import asyncio
import json
import os
import gzip
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

DEFAULT_UPSTREAM_BASE_URL = os.getenv("LLMUX_UPSTREAM_BASE_URL", "http://192.168.50.90:8082").rstrip("/")

# Optional Gemini OpenAI-compatible proxy upstream (e.g., http://172.20.0.1:8083).
GEMINI_UPSTREAM_BASE_URL = (os.getenv("GEMINI_UPSTREAM_BASE_URL") or "").rstrip("/")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Optional Z.AI (GLM) OpenAI-compatible proxy upstream.
# Z.AI uses a different base path than /v1 by default.
ZAI_UPSTREAM_BASE_URL = (os.getenv("ZAI_UPSTREAM_BASE_URL") or "").rstrip("/")  # e.g. https://api.z.ai/api/paas/v4
ZAI_CODING_UPSTREAM_BASE_URL = (os.getenv("ZAI_CODING_UPSTREAM_BASE_URL") or "").rstrip("/")  # e.g. https://api.z.ai/api/coding/paas/v4
ZAI_API_KEY = os.getenv("ZAI_API_KEY")
ZAI_CHAT_PATH = os.getenv("ZAI_CHAT_PATH", "/chat/completions")
ZAI_MODELS_PATH = os.getenv("ZAI_MODELS_PATH", "/models")
ZAI_MODEL_IDS = [m.strip() for m in (os.getenv("ZAI_MODEL_IDS") or "glm-4.7").split(",") if m.strip()]
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLMUX_SHIM_TIMEOUT_SECONDS", "300"))

app = FastAPI(title="LLMux OpenAI Shim", version="1.0.0")


def _hop_by_hop_header_names() -> set[str]:
    return {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }


def _forward_headers(request: Request) -> Dict[str, str]:
    blocked = _hop_by_hop_header_names()
    out: Dict[str, str] = {}
    for k, v in request.headers.items():
        lk = k.lower()
        if lk in blocked:
            continue
        out[k] = v
    return out


def _is_gemini_model(model: Optional[str]) -> bool:
    if not model:
        return False
    return model.startswith("gemini-") or model.startswith("gemma-") or model.startswith("text-embedding") or "embedding" in model


def _is_glm_model(model: Optional[str]) -> bool:
    if not model:
        return False
    return model.startswith("glm-") or model.startswith("glm_") or model.startswith("glm")


def _select_upstream(model: Optional[str]) -> str:
    if GEMINI_UPSTREAM_BASE_URL and _is_gemini_model(model):
        return GEMINI_UPSTREAM_BASE_URL
    if ZAI_CODING_UPSTREAM_BASE_URL and _is_glm_model(model):
        # Z.AI recommends using the coding endpoint for GLM coding plan; in practice it also works for general chat.
        return ZAI_CODING_UPSTREAM_BASE_URL
    if ZAI_UPSTREAM_BASE_URL and _is_glm_model(model):
        return ZAI_UPSTREAM_BASE_URL
    return DEFAULT_UPSTREAM_BASE_URL


def _apply_upstream_auth(headers: Dict[str, str], upstream_base_url: str, model: Optional[str]) -> Dict[str, str]:
    # For Gemini upstream, override Authorization with Gemini key if available.
    if GEMINI_UPSTREAM_BASE_URL and upstream_base_url == GEMINI_UPSTREAM_BASE_URL and _is_gemini_model(model):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set for Gemini upstream routing")
        headers = dict(headers)
        headers["Authorization"] = f"Bearer {GEMINI_API_KEY}"
        # Gemini proxy (Google frontends) may advertise gzip but return non-gzip bytes; avoid decode errors.
        headers["Accept-Encoding"] = "identity"
        return headers

    # For Z.AI upstream, override Authorization with ZAI key if available.
    if (ZAI_UPSTREAM_BASE_URL or ZAI_CODING_UPSTREAM_BASE_URL) and upstream_base_url in {
        ZAI_UPSTREAM_BASE_URL,
        ZAI_CODING_UPSTREAM_BASE_URL,
    } and _is_glm_model(model):
        if not ZAI_API_KEY:
            raise RuntimeError("ZAI_API_KEY is not set for Z.AI upstream routing")
        headers = dict(headers)
        for key in list(headers.keys()):
            if key.lower() in {"authorization", "accept-encoding", "connection"}:
                del headers[key]
        headers["Authorization"] = f"Bearer {ZAI_API_KEY}"
        headers["Accept-Language"] = "en-US,en"
        headers["Accept-Encoding"] = "identity"
        headers["Connection"] = "keep-alive"
    return headers


def _models_url_for_upstream(upstream_base_url: str) -> str:
    if upstream_base_url in {ZAI_UPSTREAM_BASE_URL, ZAI_CODING_UPSTREAM_BASE_URL}:
        return f"{upstream_base_url}{ZAI_MODELS_PATH}"
    return f"{upstream_base_url}/v1/models"


def _chat_url_for_upstream(upstream_base_url: str) -> str:
    if upstream_base_url in {ZAI_UPSTREAM_BASE_URL, ZAI_CODING_UPSTREAM_BASE_URL}:
        return f"{upstream_base_url}{ZAI_CHAT_PATH}"
    return f"{upstream_base_url}/v1/chat/completions"


def _upstream_requires_stream(upstream_base_url: str) -> bool:
    # LLMux/ChatGPT proxy and Z.AI require stream=true and return SSE format
    if upstream_base_url == DEFAULT_UPSTREAM_BASE_URL:
        return True
    # Z.AI upstreams also support streaming in SSE format
    if ZAI_UPSTREAM_BASE_URL and upstream_base_url == ZAI_UPSTREAM_BASE_URL:
        return True
    if ZAI_CODING_UPSTREAM_BASE_URL and upstream_base_url == ZAI_CODING_UPSTREAM_BASE_URL:
        return True
    return False


async def _iter_sse_lines(byte_iter: AsyncIterator[bytes]) -> AsyncIterator[str]:
    buffer = b""
    async for chunk in byte_iter:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            yield line.decode("utf-8", errors="replace").rstrip("\r")
    if buffer:
        yield buffer.decode("utf-8", errors="replace").rstrip("\r")


async def _read_raw(response: httpx.Response) -> bytes:
    chunks: list[bytes] = []
    async for part in response.aiter_raw():
        chunks.append(part)
    return b"".join(chunks)


def _maybe_decompress(raw: bytes) -> bytes:
    # Some upstreams advertise gzip inconsistently. Only decompress if the payload
    # looks like gzip (0x1f 0x8b).
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        return gzip.decompress(raw)
    return raw



def _merge_tool_calls(acc: Dict[str, Any], delta: Dict[str, Any]) -> None:
    # OpenAI streaming tool_calls are incremental; arguments are streamed as concatenated strings.
    if "id" in delta and not acc.get("id"):
        acc["id"] = delta["id"]
    if "type" in delta and not acc.get("type"):
        acc["type"] = delta["type"]
    if "function" in delta:
        acc.setdefault("function", {})
        fn_acc = acc["function"]
        fn_delta = delta["function"] or {}
        if "name" in fn_delta and not fn_acc.get("name"):
            fn_acc["name"] = fn_delta["name"]
        if "arguments" in fn_delta:
            fn_acc["arguments"] = (fn_acc.get("arguments") or "") + (fn_delta.get("arguments") or "")


async def _buffer_openai_stream_to_json(stream_bytes: AsyncIterator[bytes]) -> Dict[str, Any]:
    content_parts: List[str] = []
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    created: Optional[int] = None
    response_id: Optional[str] = None
    usage: Optional[dict] = None

    # tool_calls are streamed as a list under delta.tool_calls; keep per-index accumulators
    tool_calls_by_index: Dict[int, Dict[str, Any]] = {}

    async for line in _iter_sse_lines(stream_bytes):
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except Exception:
            continue

        response_id = response_id or event.get("id")
        model = model or event.get("model")
        created = created or event.get("created")
        if usage is None and isinstance(event.get("usage"), dict):
            usage = event["usage"]

        choices = event.get("choices") or []
        if not choices:
            continue
        choice0 = choices[0] or {}
        finish_reason = choice0.get("finish_reason") or finish_reason
        delta = choice0.get("delta") or {}

        if isinstance(delta.get("content"), str):
            content_parts.append(delta["content"])

        if isinstance(delta.get("tool_calls"), list):
            for tc in delta["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                idx = tc.get("index", 0)
                acc = tool_calls_by_index.setdefault(idx, {})
                _merge_tool_calls(acc, tc)

    tool_calls: Optional[list] = None
    if tool_calls_by_index:
        tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]

    message: Dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls

    # Letta expects `usage` to be present and a dict (its contemplate schema requires it).
    # If upstream doesn't provide usage, synthesize zeros.
    result: Dict[str, Any] = {
        "id": response_id or "chatcmpl-shim",
        "object": "chat.completion",
        "created": created or 0,
        "model": model or "unknown",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or "stop",
                "logprobs": None,
            }
        ],
        "usage": usage
        if usage is not None
        else {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        },
    }
    return result


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "default_upstream": DEFAULT_UPSTREAM_BASE_URL,
        "gemini_upstream": GEMINI_UPSTREAM_BASE_URL or None,
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "zai_upstream": ZAI_UPSTREAM_BASE_URL or None,
        "zai_coding_upstream": ZAI_CODING_UPSTREAM_BASE_URL or None,
        "zai_api_key_configured": bool(ZAI_API_KEY),
    }


@app.get("/v1/models")
async def list_models(request: Request):
    headers = _forward_headers(request)
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        async with client.stream("GET", _models_url_for_upstream(DEFAULT_UPSTREAM_BASE_URL), headers=headers) as r:
            default_resp = r
            default_raw = await _read_raw(r)

        if not GEMINI_UPSTREAM_BASE_URL and not ZAI_UPSTREAM_BASE_URL:
            return Response(
                content=default_raw,
                status_code=default_resp.status_code,
                media_type=default_resp.headers.get("content-type", "application/json"),
            )

        extra_responses: list[httpx.Response] = []
        extra_raw: list[bytes] = []
        if GEMINI_UPSTREAM_BASE_URL:
            try:
                gemini_headers = _apply_upstream_auth(headers, GEMINI_UPSTREAM_BASE_URL, "gemini-2.0-flash")
                async with client.stream("GET", _models_url_for_upstream(GEMINI_UPSTREAM_BASE_URL), headers=gemini_headers) as gr:
                    extra_responses.append(gr)
                    extra_raw.append(await _read_raw(gr))
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": {"message": str(e)}})

        if ZAI_UPSTREAM_BASE_URL and ZAI_API_KEY:
            try:
                zai_headers = _apply_upstream_auth(headers, ZAI_UPSTREAM_BASE_URL, "glm-4.7")
                async with client.stream("GET", _models_url_for_upstream(ZAI_UPSTREAM_BASE_URL), headers=zai_headers) as zr:
                    if zr.status_code == 200:
                        extra_responses.append(zr)
                        extra_raw.append(await _read_raw(zr))
            except Exception:
                pass

    # Merge model lists (best-effort); if parsing fails, fall back to default response.
    try:
        default_json = json.loads(_maybe_decompress(default_raw).decode("utf-8"))
        merged: Dict[str, Any] = {"object": "list", "data": []}
        seen: set[str] = set()
        payloads = [default_json] + [json.loads(_maybe_decompress(b).decode("utf-8")) for b in extra_raw]
        for payload in payloads:
            for item in (payload.get("data") or []):
                mid = item.get("id")
                if not isinstance(mid, str) or mid in seen:
                    continue
                seen.add(mid)
                merged["data"].append(item)

        # If Z.AI is configured (or you want GLM models to appear), optionally inject known IDs.
        # This keeps GLM selectable even if Z.AI doesn't expose a /models endpoint.
        for mid in ZAI_MODEL_IDS:
            if mid in seen:
                continue
            seen.add(mid)
            merged["data"].append({"id": mid, "object": "model", "created": 0, "owned_by": "zai"})
        return JSONResponse(status_code=200, content=merged)
    except Exception:
        return Response(
            content=default_raw,
            status_code=default_resp.status_code,
            media_type=default_resp.headers.get("content-type", "application/json"),
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON body"}})

    client_wants_stream = bool(body.get("stream"))
    requested_model = body.get("model") if isinstance(body.get("model"), str) else None
    upstream_base_url = _select_upstream(requested_model)
    upstream_body = dict(body)
    
    if _upstream_requires_stream(upstream_base_url):
        upstream_body["stream"] = True
    else:
        upstream_body["stream"] = bool(body.get("stream"))

    try:
        headers = _apply_upstream_auth(_forward_headers(request), upstream_base_url, requested_model)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": {"message": str(e)}})
    url = _chat_url_for_upstream(upstream_base_url)

    if client_wants_stream:
        async def _shimmed_stream() -> AsyncIterator[bytes]:
            saw_usage = False
            last_id: Optional[str] = None
            last_model: Optional[str] = None
            last_created: Optional[int] = None

            client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)
            try:
                async with client.stream("POST", url, headers=headers, json=upstream_body) as r:
                    if r.status_code >= 400:
                        raw = await _read_raw(r)
                        yield raw
                        return
                    if not _upstream_requires_stream(upstream_base_url):
                        try:
                            async for b in r.aiter_raw():
                                yield b
                        except Exception:
                            pass
                        return

                    async for line in _iter_sse_lines(r.aiter_raw()):
                        yield (line + "\n").encode("utf-8")

                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            event = json.loads(data)
                        except Exception:
                            continue
                        last_id = last_id or event.get("id")
                        last_model = last_model or event.get("model")
                        last_created = last_created or event.get("created")
                        if isinstance(event.get("usage"), dict):
                            saw_usage = True
            finally:
                await client.aclose()

            # If upstream omitted usage (common for proxies), emit a final usage-only chunk.
            if not saw_usage:
                usage_chunk = {
                    "id": last_id or "chatcmpl-shim",
                    "object": "chat.completion.chunk",
                    "created": last_created or 0,
                    "model": last_model or upstream_body.get("model") or "unknown",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "prompt_tokens_details": None,
                        "completion_tokens_details": None,
                    },
                }
                yield ("data: " + json.dumps(usage_chunk) + "\n\n").encode("utf-8")

        return StreamingResponse(_shimmed_stream(), media_type="text/event-stream; charset=utf-8")

    # Non-streaming
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        if _upstream_requires_stream(upstream_base_url):
            async with client.stream("POST", url, headers=headers, json=upstream_body) as r:
                if r.status_code >= 400:
                    raw = await _read_raw(r)
                    try:
                        return JSONResponse(status_code=r.status_code, content=json.loads(raw.decode("utf-8")))
                    except Exception:
                        return Response(content=raw, status_code=r.status_code, media_type=r.headers.get("content-type"))

                data = await _buffer_openai_stream_to_json(r.aiter_raw())
                return JSONResponse(status_code=200, content=data)

        # For upstreams that support non-streaming, pass through the JSON response as-is.
        resp = await client.post(url, headers=headers, json=upstream_body)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def passthrough(path: str, request: Request):
    # Best-effort passthrough for any other endpoints.
    model = request.query_params.get("model")
    upstream_base_url = _select_upstream(model)
    try:
        headers = _apply_upstream_auth(_forward_headers(request), upstream_base_url, model)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": {"message": str(e)}})
    url = f"{upstream_base_url}/{path}"
    method = request.method.upper()
    params = dict(request.query_params)
    content = await request.body()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        upstream = await client.request(method, url, headers=headers, params=params, content=content)

    media_type = upstream.headers.get("content-type")
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8090")))
