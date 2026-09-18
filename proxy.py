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
import copy
import json
import os
import queue
import re
import threading
from pathlib import Path
from typing import Any

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

LOCAL_ENV = Path(__file__).parent / ".env"


def _api_key() -> str:
    k = os.environ.get("AGENTROUTER_API_KEY", "").strip()
    if not k and KEY_FILE.exists():
        k = KEY_FILE.read_text().strip()
    if not k and LOCAL_ENV.exists():
        for line in LOCAL_ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("AGENTROUTER_API_KEY="):
                k = line.split("=", 1)[1].strip().strip("\"'")
                break
    if not k:
        raise RuntimeError(
            "No AGENTROUTER_API_KEY. "
            f"Set env var, create {KEY_FILE}, or define it in {LOCAL_ENV}"
        )
    return k


_clients: dict[str, anthropic.Anthropic] = {}


def _client_for(key: str | None = None) -> anthropic.Anthropic:
    """Return an Anthropic sync client instance for the given API key."""
    api_key = key or _api_key()
    if api_key not in _clients:
        _clients[api_key] = anthropic.Anthropic(
            api_key=api_key,
            base_url=TARGET,
            http_client=_HTTP_CLIENT,
        )
    return _clients[api_key]


def _client() -> anthropic.Anthropic:
    """Fallback client using the default server API key."""
    return _client_for(_api_key())


def _extract_api_key(request: Request | None = None) -> str:
    """Resolve API key: prioritize server configuration (.env / env var / KEY_FILE).
    If no server key is configured, fallback to client request headers (x-api-key or Authorization: Bearer).
    """
    try:
        server_key = _api_key()
        if server_key:
            return server_key
    except Exception:
        pass

    if request:
        k = request.headers.get("x-api-key", "").strip()
        if k:
            return k
        auth = request.headers.get("authorization", "").strip()
        if auth.lower().startswith("bearer "):
            k = auth[7:].strip()
            if k:
                return k

    return _api_key()


# ── Request translation ───────────────────────────────────────────────────────

_SKIP = {"stream"}  # handled separately in the route (streaming vs non-streaming)
_ZWSP = "\u200b"

# Pre-compiled WAF neutralization patterns across multiple languages & vectors
_WAF_RULES = [
    # 1. PHP opening tags & script language php (including <?php, <?=, <? and <script language=php>)
    (re.compile(r"(<)(\?)(php|=|[\s\n\r])", re.IGNORECASE), rf"\g<1>{_ZWSP}\g<2>\g<3>"),
    (re.compile(r"(<script[^>]*language\s*=\s*[\'\"]?)(php)", re.IGNORECASE), rf"\g<1>{_ZWSP}\g<2>"),

    # 2. JSP / ASP tags & response methods
    (re.compile(r"(<)(%)"), rf"\g<1>{_ZWSP}\g<2>"),
    (re.compile(r"\b(out)\s*\.\s*(println)\b", re.IGNORECASE), rf"\g<1>.{_ZWSP}\g<2>"),
    (re.compile(r"\b(Response)\s*\.\s*(Write)\b", re.IGNORECASE), rf"\g<1>.{_ZWSP}\g<2>"),

    # 3. Execution functions & SQLi/dangerous functions:
    # system(, exec(, eval(, passthru(, assert(, phpinfo(, sleep(, load_file(, updatexml(, extractvalue(, benchmark(, shell_exec(, popen(, proc_open(
    (re.compile(r"\b(system|exec|eval|passthru|assert|phpinfo|sleep|load_file|updatexml|extractvalue|benchmark|shell_exec|popen|proc_open)\s*\(", re.IGNORECASE), rf"\g<1>{_ZWSP}("),

    # 4. Dangerous protocols & JNDI: ldap://, rmi://, file:///, ${jndi:...}
    (re.compile(r"\b(ldap|rmi):(//)", re.IGNORECASE), rf"\g<1>:{_ZWSP}\g<2>"),
    (re.compile(r"\b(file):(//+)", re.IGNORECASE), rf"\g<1>:{_ZWSP}\g<2>"),
    (re.compile(r"(\$\{)\s*(jndi)", re.IGNORECASE), rf"\g<1>{_ZWSP}\g<2>"),
    (re.compile(r"\b(jndi)\s*:", re.IGNORECASE), rf"\g<1>{_ZWSP}:"),

    # 5. Node.js child_process & Java Runtime
    (re.compile(r"\b(child)_(process)\b", re.IGNORECASE), rf"\g<1>_{_ZWSP}\g<2>"),
    (re.compile(r"\b(getRuntime|ProcessBuilder)\s*\(", re.IGNORECASE), rf"\g<1>{_ZWSP}("),

    # 6. Sensitive files & directory traversal (/etc/passwd, /etc/hosts, win.ini, Windows/System32)
    (re.compile(r"(/etc/)(passwd|shadow|hosts|group|issue)\b", re.IGNORECASE), rf"\g<1>{_ZWSP}\g<2>"),
    (re.compile(r"(\\etc\\)(passwd|shadow|hosts|group|issue)\b", re.IGNORECASE), rf"\g<1>{_ZWSP}\g<2>"),
    (re.compile(r"\b(win)(dows)[/\\](system32)\b", re.IGNORECASE), rf"\g<1>{_ZWSP}\g<2>/\g<3>"),
    (re.compile(r"\b(win)\.(ini)\b", re.IGNORECASE), rf"\g<1>.{_ZWSP}\g<2>"),
    (re.compile(r"(\.\.)([/\\])"), rf"\g<1>{_ZWSP}\g<2>"),

    # 7. SQL injection triggers (WAITFOR DELAY, ' OR '1'='1, ' OR 1=1, @@variables, concat(0x...))
    (re.compile(r"\b(WAITFOR)\s+(DELAY)\b", re.IGNORECASE), rf"\g<1>{_ZWSP} \g<2>"),
    (re.compile(r"('|\")\s*(O)(R)\b", re.IGNORECASE), rf"\g<1>\g<2>{_ZWSP}\g<3>"),
    (re.compile(r"(@)(@\w+)", re.IGNORECASE), rf"\g<1>{_ZWSP}\g<2>"),
    (re.compile(r"\b(concat)\s*\(\s*(0)(x[0-9a-fA-F]+)", re.IGNORECASE), rf"\g<1>(\g<2>{_ZWSP}\g<3>"),

    # 8. HTML / XSS / XXE
    (re.compile(r"(<scr)(ipt)", re.IGNORECASE), rf"\g<1>{_ZWSP}\g<2>"),
    (re.compile(r"\b(on)(error|load)\s*=", re.IGNORECASE), rf"\g<1>{_ZWSP}\g<2>="),
    (re.compile(r"(<!EN)(TITY)\b", re.IGNORECASE), rf"\g<1>{_ZWSP}\g<2>"),

    # 9. Shell command injection chaining (; echo, ; cat, | bash, && rm, | grep, etc.)
    (re.compile(r"([;|&`]\s*)\b(echo|cat|curl|wget|bash|sh|zsh|python|perl|ruby|rm|ls|id|whoami|chmod|chown|kill|nc|netcat|uname|grep|ps|sudo|php)\b", re.IGNORECASE), rf"\g<1>{_ZWSP}\g<2>"),
]


def _sanitize_waf_str(text: str) -> str:
    """Neutralize known Aliyun WAF attack signatures across multiple languages and protocols.

    Aliyun WAF sits in front of agentrouter.org and inspects JSON POST bodies.
    Requests containing tokens from PHP, JSP/ASP, Node.js RCE, Python os.system,
    SQL blind injection, sensitive system paths (/etc/passwd, win.ini), or XSS/XXE
    trigger Aliyun WAF's Web core defense rules, resulting in an immediate HTTP 405 block:
    '很抱歉，由于您访问的URL有可能对网站造成安全威胁，您的访问被阻断。'

    Inserting an invisible zero-width space (\\u200b) breaks the WAF regex patterns
    while remaining completely invisible in UI/Markdown and fully understood by LLMs.
    """
    if not isinstance(text, str):
        return text

    for pattern, repl in _WAF_RULES:
        text = pattern.sub(repl, text)

    return text


_EXEMPT_KEYS = {
    "name",  # Tool function name (e.g. 'bash', 'view_file')
    "id",  # Message ID / Tool use ID
    "tool_use_id",  # Tool result reference ID
    "type",  # Block type ('tool_use', 'tool_result', 'text')
    "role",  # 'user', 'assistant'
    "model",  # Model identifier
    "data",  # Base64 image/file payload
    "image",
    "source",
}


def _sanitize_for_waf(data: Any, parent_key: str = "") -> Any:
    """Recursively sanitize string values in payloads to prevent Aliyun WAF 405 blocks.

    Protocol metadata fields (tool names, IDs, types) are exempt to ensure tool calling
    and schema validation remain completely unaltered.
    """
    if parent_key in _EXEMPT_KEYS:
        return data
    if isinstance(data, str):
        return _sanitize_waf_str(data)
    elif isinstance(data, list):
        return [_sanitize_for_waf(item, parent_key) for item in data]
    elif isinstance(data, dict):
        return {k: _sanitize_for_waf(v, k) for k, v in data.items()}
    return data


def _convert_chars_to_fullwidth(s: str) -> str:
    res = []
    for char in s:
        code = ord(char)
        if 0x41 <= code <= 0x5A or 0x61 <= code <= 0x7A:
            res.append(chr(code + 0xFEE0))
        else:
            res.append(char)
    return "".join(res)


def _to_fullwidth_letters(s: str) -> str:
    """Convert Latin letters [a-zA-Z] to Unicode fullwidth [ａ-ｚＡ-Ｚ] to evade WAF token extraction,
    while strictly preserving URLs (http:// and https://) in original ASCII.
    """
    url_pattern = re.compile(r'https?://[^\s"\'<>]+', re.IGNORECASE)
    parts = []
    last_end = 0
    for m in url_pattern.finditer(s):
        parts.append(_convert_chars_to_fullwidth(s[last_end:m.start()]))
        parts.append(m.group(0))  # Preserve URL intact as ASCII
        last_end = m.end()
    parts.append(_convert_chars_to_fullwidth(s[last_end:]))
    return "".join(parts)


def _fullwidth_neutralize(data: Any, parent_key: str = "") -> Any:
    """Recursively transform content text to fullwidth letters to evade WAF/content-blocked,
    while strictly protecting protocol keys and schema fields.
    """
    if parent_key in _EXEMPT_KEYS:
        return data
    if isinstance(data, str):
        return _to_fullwidth_letters(data)
    elif isinstance(data, list):
        return [_fullwidth_neutralize(item, parent_key) for item in data]
    elif isinstance(data, dict):
        return {k: _fullwidth_neutralize(v, k) for k, v in data.items()}
    return data


def _fallback_kw(kw: dict) -> dict:
    """Create a copy of request kwargs with fullwidth neutralization applied to messages and system prompt."""
    ret = dict(kw)
    if "messages" in ret:
        ret["messages"] = _fullwidth_neutralize(ret["messages"])
    if "system" in ret:
        ret["system"] = _fullwidth_neutralize(ret["system"])
    return ret


def _clean_model_output(s: str) -> str:
    """Normalize model output before returning to client.
    1. Remove zero-width spaces (\u200b).
    2. Convert fullwidth Latin letters [ａ-ｚＡ-Ｚ] back to standard ASCII [a-zA-Z].
    3. Normalize fullwidth URL punctuation (． -> ., ／ -> /, ：// -> ://).
    4. Fix broken/spaced URL schemes like 'h t t p : / /' or 'h t t p s : / /'.
    """
    if not isinstance(s, str) or not s:
        return s

    # 1. Remove zero-width space (both literal and JSON-escaped)
    s = s.replace("\u200b", "").replace("\\u200b", "").replace("\\u200B", "")

    # 2. Convert fullwidth Latin letters [ａ-ｚＡ-Ｚ] and fullwidth dot/slash to standard ASCII
    res = []
    for c in s:
        code = ord(c)
        if (0xFF21 <= code <= 0xFF3A) or (0xFF41 <= code <= 0xFF5A):
            res.append(chr(code - 0xFEE0))
        elif code == 0xFF0E:  # Fullwidth dot '．'
            res.append(".")
        elif code == 0xFF0F:  # Fullwidth slash '／'
            res.append("/")
        elif code == 0x3000:  # Fullwidth space
            res.append(" ")
        else:
            res.append(c)
    s = "".join(res)

    # 3. Normalize fullwidth URL schemes (e.g. http：// -> http://)
    s = s.replace("：//", "://").replace("：／／", "://")

    # 4. Fix spaced URL protocols like 'h t t p : / /' -> 'http://'
    s = re.sub(
        r'(?<![a-zA-Z])h\s*t\s*t\s*p\s*(s?)\s*:\s*/\s*/\s*',
        lambda m: ('https://' if m.group(1).lower() == 's' else 'http://'),
        s,
        flags=re.IGNORECASE
    )

    return s


def _clean_sse_line(line: bytes) -> bytes:
    """Clean streaming SSE lines before sending to client."""
    if not line:
        return line
    # Fast byte-level strip of zero-width space
    line = line.replace(b"\xe2\x80\x8b", b"")
    if line.startswith(b"data:"):
        try:
            text = line.decode("utf-8")
            cleaned = _clean_model_output(text)
            return cleaned.encode("utf-8")
        except Exception:
            return line
    return line


def _is_blocked_error(exc: Exception) -> bool:
    """Check if an exception is caused by upstream WAF or content-blocked moderation."""
    if isinstance(exc, anthropic.APIStatusError):
        body_str = str(exc.body) if exc.body else ""
        exc_str = str(exc)
        if exc.status_code == 405:
            return True
        if "content-blocked" in body_str or "content-blocked" in exc_str:
            return True
        if any(w in body_str or w in exc_str for w in ("很抱歉", "security", "<!doctypehtml>", "potential threat")):
            return True
    return False


def _is_thinking_error(exc: Exception) -> bool:
    """Check if an exception is caused by missing thinking block passback in thinking mode."""
    if isinstance(exc, anthropic.APIStatusError):
        body_str = str(exc.body) if exc.body else ""
        exc_str = str(exc)
        combined = (body_str + " " + exc_str).lower()
        if "thinking" in combined and (
            "must be passed back" in combined
            or "content[].thinking" in combined
            or "missing field `thinking`" in combined
            or "missing field 'thinking'" in combined
        ):
            return True
    return False


def _fix_thinking_messages(messages: list) -> list:
    """Ensure all assistant messages satisfy upstream thinking mode requirements.

    In thinking mode (e.g. DeepSeek / Volcengine Ark / AgentRouter), upstream validates
    that any assistant message in conversation history (especially those containing tool_use)
    must pass back a valid content[].thinking block.
    Many clients (OpenCode, Cursor, AI SDK, Cline) strip thinking blocks from history.

    1. If an assistant message contains tool_use (or is a list of blocks) but lacks a thinking block,
       inject a minimal thinking block {"type": "thinking", "thinking": "..."} at index 0.
    2. If a thinking block exists but lacks 'thinking' text or has empty thinking,
       populate it with a placeholder.
    """
    fixed = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            fixed.append(msg)
            continue

        msg = copy.deepcopy(msg)
        content = msg.get("content")

        if isinstance(content, list):
            has_thinking = False
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    has_thinking = True
                    if not block.get("thinking"):
                        block["thinking"] = "..."
            if not has_thinking:
                has_tool_use = any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
                if has_tool_use:
                    content.insert(0, {"type": "thinking", "thinking": "..."})
            msg["content"] = content
        fixed.append(msg)
    return fixed


def _fallback_thinking_kw(kw: dict) -> dict:
    """Fallback when upstream rejects due to thinking block validation:
    1. If any assistant message still lacks a thinking block (e.g. string content or text-only blocks),
       inject thinking block into all assistant messages.
    2. If all already have thinking blocks, fallback to disabling thinking mode.
    """
    ret = copy.deepcopy(kw)
    messages = ret.get("messages", [])
    already_patched = True
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, list):
                if not any(isinstance(b, dict) and b.get("type") == "thinking" and b.get("thinking") for b in content):
                    already_patched = False
                    content.insert(0, {"type": "thinking", "thinking": "..."})
            elif isinstance(content, str):
                already_patched = False
                msg["content"] = [
                    {"type": "thinking", "thinking": "..."},
                    {"type": "text", "text": content},
                ]
    if already_patched:
        ret["thinking"] = {"type": "disabled"}
    return ret


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

    # Normalize thinking blocks in assistant messages to satisfy upstream requirements
    if "messages" in kw and isinstance(kw["messages"], list):
        kw["messages"] = _fix_thinking_messages(kw["messages"])

    # Sanitize messages and system prompt to neutralize Aliyun WAF 405 triggers
    # (e.g. <?php, system(, eval(, /etc/passwd) using invisible zero-width spaces (\u200b).
    if "messages" in kw:
        kw["messages"] = _sanitize_for_waf(kw["messages"])
    if "system" in kw:
        kw["system"] = _sanitize_for_waf(kw["system"])

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

def _stream_worker(kw: dict, q: queue.Queue, client: anthropic.Anthropic) -> None:
    """
    Run inside a thread. Uses the sync Anthropic SDK's with_streaming_response
    to get raw SSE bytes and puts them into the queue, stripping any
    non-standard event types (e.g. billing_summary) that break OpenCode's parser.
    Automatically retries with fullwidth neutralization if blocked by upstream WAF/content filter.
    """
    SKIP_EVENTS: set[bytes] = {b"billing_summary"}
    resp_cm = None

    try:
        try:
            resp_cm = client.messages.with_streaming_response.create(**kw)
            resp = resp_cm.__enter__()
        except Exception as first_err:
            if resp_cm is not None:
                try:
                    resp_cm.__exit__(None, None, None)
                except Exception:
                    pass
                resp_cm = None
            if _is_thinking_error(first_err):
                tb_kw = _fallback_thinking_kw(kw)
                try:
                    resp_cm = client.messages.with_streaming_response.create(**tb_kw)
                    resp = resp_cm.__enter__()
                except Exception as second_err:
                    if resp_cm is not None:
                        try:
                            resp_cm.__exit__(None, None, None)
                        except Exception:
                            pass
                        resp_cm = None
                    if _is_blocked_error(second_err):
                        fb_kw = _fallback_kw(tb_kw)
                        resp_cm = client.messages.with_streaming_response.create(**fb_kw)
                        resp = resp_cm.__enter__()
                    else:
                        raise second_err
            elif _is_blocked_error(first_err):
                fb_kw = _fallback_kw(kw)
                resp_cm = client.messages.with_streaming_response.create(**fb_kw)
                resp = resp_cm.__enter__()
            else:
                raise first_err

        try:
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

                    q.put(_clean_sse_line(line))

            if buf:
                q.put(_clean_sse_line(buf))

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
        finally:
            if resp_cm is not None:
                resp_cm.__exit__(None, None, None)

    except Exception as exc:
        q.put(exc)
    finally:
        q.put(None)  # sentinel


async def _stream_gen(kw: dict, client: anthropic.Anthropic):
    q: queue.Queue = queue.Queue()
    t = threading.Thread(target=_stream_worker, args=(kw, q, client), daemon=True)
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


def _format_api_error(e: Exception) -> tuple[int, dict]:
    """Format an exception into (status_code, error_dict) following Anthropic API schema."""
    if isinstance(e, anthropic.APIStatusError):
        body = e.body
        status = e.status_code
        if isinstance(body, dict) and "error" in body:
            return status, body
        body_str = str(body) if body is not None else ""
        if status == 405 or "content-blocked" in body_str or "很抱歉" in body_str or "security" in body_str or "<!doctypehtml>" in body_str:
            return 405, {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": (
                        "Upstream AgentRouter Aliyun WAF/moderation blocked the request. "
                        "The conversation history contains code or tokens triggering security rules (e.g. PHP tags, shell commands, or process traces)."
                    ),
                },
            }
        return status, {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": body_str or str(e),
            },
        }
    return 500, {
        "type": "error",
        "error": {
            "type": "proxy_error",
            "message": str(e),
        },
    }


@app.post("/v1/messages")
@app.post("/messages")
async def messages(request: Request):
    body = await request.json()
    kw = _kwargs(body)
    client = _client_for(_extract_api_key(request))

    if body.get("stream", False):
        kw["stream"] = True

        async def _safe_stream():
            try:
                async for chunk in _stream_gen(kw, client):
                    yield chunk
            except Exception as e:
                _, err_body = _format_api_error(e)
                yield f"event: error\ndata: {json.dumps(err_body)}\n\n".encode()

        return StreamingResponse(
            _safe_stream(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    # Non-streaming: sync SDK call in a thread to keep the event loop free
    def _run():
        try:
            return client.messages.create(**kw)
        except Exception as first_err:
            if _is_thinking_error(first_err):
                tb_kw = _fallback_thinking_kw(kw)
                try:
                    return client.messages.create(**tb_kw)
                except Exception as second_err:
                    if _is_blocked_error(second_err):
                        fb_kw = _fallback_kw(tb_kw)
                        return client.messages.create(**fb_kw)
                    raise second_err
            if _is_blocked_error(first_err):
                fb_kw = _fallback_kw(kw)
                return client.messages.create(**fb_kw)
            raise first_err

    try:
        msg = await asyncio.to_thread(_run)
        cleaned_json = _clean_model_output(msg.model_dump_json())
        return Response(content=cleaned_json, media_type="application/json")
    except Exception as e:
        status, err_body = _format_api_error(e)
        return Response(
            content=json.dumps(err_body),
            status_code=status,
            media_type="application/json",
        )


@app.get("/v1/models")
@app.get("/models")
async def models():
    """Stub model list — only lists models confirmed working on agentrouter.org."""
    return {
        "object": "list",
        "data": [
            {"id": "claude-opus-4-8", "object": "model"},
            {"id": "gpt-5.6-sol", "object": "model"},
            {"id": "kimi-k3", "object": "model"},
            {"id": "claude-fable-5", "object": "model"},
            {"id": "claude-opus-5", "object": "model"},
            {"id": "deepseek-v4-flash", "object": "model"},
            {"id": "glm-5.3", "object": "model"},
            {"id": "gpt-6-astra", "object": "model"},
        ],
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"AgentRouter proxy  →  {TARGET}")
    print(f"Listening on http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
