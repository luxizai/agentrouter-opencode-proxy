#!/usr/bin/env python3
"""
agentrouter-proxy: thin reverse-proxy for agentrouter.org.

AgentRouter's Aliyun WAF fingerprints TLS handshakes AND inspects
Anthropic SDK-specific headers (x-stainless-*, user-agent). Raw httpx
is blocked, and the AsyncAnthropic client is also rejected because its
asyncio SSL implementation produces a different TLS fingerprint.
Only requests made through the Python sync `anthropic` SDK pass the check.

This proxy keeps a single sync `anthropic.Anthropic` client (shared
connection pool, short keepalive_expiry to prevent zombie connections
when the upstream load balancer silently closes idle sockets).
Both the non-streaming and streaming paths offload to a thread via
asyncio.to_thread / a thread+queue so the FastAPI event loop stays free.

Usage:
    python proxy.py           # reads key from ~/.config/opencode/api_keys/AGENT_ROUTER_API_KEY
    AGENTROUTER_API_KEY=sk-... python proxy.py
"""
import asyncio
import json
import os
import queue
import threading
from pathlib import Path

import anthropic
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

# ── Config ────────────────────────────────────────────────────────────────────

TARGET = "https://agentrouter.org"
PORT = 7187
KEY_FILE = Path.home() / ".config/opencode/api_keys/AGENT_ROUTER_API_KEY"

# Seconds between received SSE chunks before aborting a streaming request.
CHUNK_TIMEOUT = 120

# agentrouter.org's load balancer drops idle connections after ~30 s.
# Setting keepalive_expiry below that threshold prevents our pool from
# trying to reuse a connection the server has already closed (zombie socket).
_HTTP_CLIENT = httpx.Client(
    limits=httpx.Limits(
        max_connections=50,
        max_keepalive_connections=5,
        keepalive_expiry=20.0,
    ),
    timeout=httpx.Timeout(connect=10, read=CHUNK_TIMEOUT, write=10, pool=5),
)


def _api_key() -> str:
    k = os.environ.get("AGENTROUTER_API_KEY", "").strip()
    if not k and KEY_FILE.exists():
        k = KEY_FILE.read_text().strip()
    if not k:
        raise RuntimeError(
            "No AGENTROUTER_API_KEY. "
            f"Set env var or create {KEY_FILE}"
        )
    return k


_anthropic_client: anthropic.Anthropic | None = None


def _client() -> anthropic.Anthropic:
    """Return the module-level sync client singleton (connection pool reused across requests)."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(
            api_key=_api_key(),
            base_url=TARGET,
            http_client=_HTTP_CLIENT,
        )
    return _anthropic_client


# ── Request translation ───────────────────────────────────────────────────────

_SKIP = {"stream"}  # handled separately in the route (streaming vs non-streaming)


def _kwargs(body: dict) -> dict:
    """Forward all fields except stream.

    Thinking is supported: when the client doesn't specify a thinking /
    effort configuration, default to adaptive thinking at high effort
    (high is also the Anthropic default effort). Client-supplied thinking
    and output_config always win, so an explicit client choice (e.g.
    effort: "xhigh") is respected rather than overridden.

    Verified against agentrouter.org for claude-opus-4-8, gpt-5.6-sol, and
    glm-5.2 — all accept these fields. The earlier note about them
    triggering the content filter was inaccurate: 405 WAF blocks are
    triggered by request-body code content (PHP/shell tokens), not by
    thinking fields.

    max_tokens defaults to 64000 when the client omits it — generous
    headroom for adaptive thinking plus output. Verified safe for both
    streaming and non-streaming against agentrouter.org.

    For OpenAI-compatible models (non-claude-*), reasoning depth is also
    signalled via the OpenAI-style `reasoning_effort` field. The Anthropic
    SDK rejects `reasoning_effort` as a direct kwarg (TypeError), so it is
    popped from the body and routed through `extra_body`. A value is always
    present on outgoing requests (default "high", mirroring effort);
    AgentRouter accepts it for gpt-*/glm-* and ignores it for Claude, so
    it is harmless to send unconditionally across all model families.
    """
    kw = {k: v for k, v in body.items() if k not in _SKIP}

    # reasoning_effort is an OpenAI-format param the SDK rejects as a direct
    # kwarg — pop it here so it never reaches messages.create(**kw) directly;
    # it is re-attached via extra_body below for every model.
    re_effort = kw.pop("reasoning_effort", None)

    if "thinking" not in kw:
        kw["thinking"] = {"type": "adaptive"}
    if "output_config" not in kw:
        kw["output_config"] = {"effort": "high"}
    if "max_tokens" not in kw:
        kw["max_tokens"] = 64000

    # Always send reasoning_effort for every model: default it to high when
    # the client didn't specify one (mirroring output_config.effort). Sent via
    # extra_body on every request — Claude models ignore it upstream, so
    # there's no downside to sending it unconditionally, and it keeps all
    # model families on the same path. Client-supplied values are passed
    # through verbatim (no clamping — xhigh/max go through as-is).
    if re_effort is None:
        oc = kw.get("output_config")
        if isinstance(oc, dict) and "effort" in oc:
            re_effort = oc["effort"]
        else:
            re_effort = "high"

    kw["extra_body"] = {"reasoning_effort": re_effort}

    return kw


# ── Streaming helper ──────────────────────────────────────────────────────────

def _stream_worker(kw: dict, q: queue.Queue) -> None:
    """
    Run inside a thread. Uses the sync Anthropic SDK's with_streaming_response
    to get raw SSE bytes and puts them into the queue, stripping any
    non-standard event types (e.g. billing_summary) that break OpenCode's parser.
    """
    SKIP_EVENTS: set[bytes] = {b"billing_summary"}

    try:
        with _client().messages.with_streaming_response.create(**kw) as resp:
            buf = b""
            skip_block = False
            # Track terminal events so we can synthesize any the upstream
            # translator omitted (see the note after the loop).
            block_open = False
            seen_message_delta = False
            seen_message_stop = False

            for raw_chunk in resp.iter_bytes(chunk_size=1024):
                buf += raw_chunk

                while True:
                    nl = buf.find(b"\n")
                    if nl == -1:
                        break
                    line = buf[: nl + 1]  # include \n
                    buf = buf[nl + 1:]
                    stripped = line.rstrip(b"\r\n")

                    if stripped.startswith(b"event:"):
                        event_name = stripped[6:].strip()
                        skip_block = event_name in SKIP_EVENTS
                        if skip_block:
                            continue
                        if event_name == b"content_block_start":
                            block_open = True
                        elif event_name == b"content_block_stop":
                            block_open = False
                        elif event_name == b"message_delta":
                            seen_message_delta = True
                        elif event_name == b"message_stop":
                            seen_message_stop = True
                    elif skip_block:
                        if stripped == b"":
                            skip_block = False  # blank line ends the event block
                        continue

                    q.put(line)

            if buf:
                q.put(buf)

            # AgentRouter's OpenAI→Anthropic SSE translator (used for non-Claude
            # models like gpt-5.6-sol / glm-5.2) ends the stream after the last
            # content_block_delta WITHOUT emitting content_block_stop /
            # message_delta / message_stop. With no message_delta there is no
            # stop_reason, so AI-SDK clients (@ai-sdk/anthropic) default the
            # finish_reason to "other" and hard-fail (Cherry Studio's
            # AI_FinishReasonError). The SDK iterator finished without raising,
            # so the connection closed cleanly and the model stopped normally —
            # synthesize the missing terminal events with stop_reason=end_turn,
            # matching what the non-streaming path returns for these models.
            if not seen_message_stop:
                if block_open:
                    q.put(b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n')
                if not seen_message_delta:
                    q.put(
                        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":0}}\n\n')
                q.put(b'event: message_stop\ndata: {"type":"message_stop"}\n\n')

    except Exception as exc:
        q.put(exc)
    finally:
        q.put(None)  # sentinel


async def _stream_gen(kw: dict):
    q: queue.Queue = queue.Queue()
    t = threading.Thread(target=_stream_worker, args=(kw, q), daemon=True)
    t.start()
    loop = asyncio.get_running_loop()
    while True:
        try:
            chunk = await asyncio.wait_for(
                loop.run_in_executor(None, q.get),
                timeout=CHUNK_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"No chunk received from agentrouter.org in {CHUNK_TIMEOUT}s — upstream stalled"
            )
        if chunk is None:
            break
        if isinstance(chunk, Exception):
            raise chunk
        yield chunk


# ── Routes ────────────────────────────────────────────────────────────────────

app = FastAPI()


@app.post("/v1/messages")
@app.post("/messages")
async def messages(request: Request):
    body = await request.json()
    kw = _kwargs(body)

    if body.get("stream", False):
        kw["stream"] = True

        async def _safe_stream():
            try:
                async for chunk in _stream_gen(kw):
                    yield chunk
            except anthropic.APIStatusError as e:
                err = json.dumps({"type": "error", "error": {"type": "api_error", "message": str(e.body)}})
                yield f"event: error\ndata: {err}\n\n".encode()
            except Exception as e:
                err = json.dumps({"type": "error", "error": {"type": "api_error", "message": str(e)}})
                yield f"event: error\ndata: {err}\n\n".encode()

        return StreamingResponse(
            _safe_stream(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    # Non-streaming: sync SDK call in a thread to keep the event loop free
    def _run():
        return _client().messages.create(**kw)

    try:
        msg = await asyncio.to_thread(_run)
        return Response(content=msg.model_dump_json(), media_type="application/json")
    except anthropic.APIStatusError as e:
        return Response(
            content=json.dumps(e.body) if e.body else b"",
            status_code=e.status_code,
            media_type="application/json",
        )
    except Exception as e:
        return Response(
            content=json.dumps({"error": {"message": str(e), "type": "proxy_error"}}),
            status_code=500,
            media_type="application/json",
        )


@app.get("/v1/models")
@app.get("/models")
async def models():
    """Stub model list — only lists models confirmed working on agentrouter.org."""
    return {
        "object": "list",
        "data": [
            {"id": "claude-opus-4-6", "object": "model"},
            {"id": "claude-opus-4-7", "object": "model"},
            {"id": "claude-opus-4-8", "object": "model"},
            {"id": "gpt-5.5", "object": "model"},
            {"id": "gpt-5.6-sol", "object": "model"},
            {"id": "glm-5.2", "object": "model"},
            {"id": "kimi-k3", "object": "model"},
            {"id": "claude-fable-5", "object": "model"},
            {"id": "claude-opus-5", "object": "model"},
        ],
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"AgentRouter proxy  →  {TARGET}")
    print(f"Listening on http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
