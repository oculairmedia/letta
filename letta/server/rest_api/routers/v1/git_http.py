"""Git HTTP Smart Protocol endpoints via dulwich (proxied).

## Why a separate dulwich server?

Dulwich's `HTTPGitApplication` is a WSGI app and relies on the WSGI `write()`
callback pattern. Starlette's `WSGIMiddleware` does not fully support this
pattern, which causes failures when mounting dulwich directly into FastAPI.

To avoid the ASGI/WSGI impedance mismatch, we run dulwich's WSGI server on a
separate local port (default: 8284) and proxy `/v1/git/*` requests to it.

Example:

    git clone http://localhost:8283/v1/git/{agent_id}/state.git

Routes (smart HTTP):
    GET  /v1/git/{agent_id}/state.git/info/refs?service=git-upload-pack
    POST /v1/git/{agent_id}/state.git/git-upload-pack
    GET  /v1/git/{agent_id}/state.git/info/refs?service=git-receive-pack
    POST /v1/git/{agent_id}/state.git/git-receive-pack

The dulwich server uses `GCSBackend` to materialize repositories from GCS on
-demand.

Post-push sync back to GCS/PostgreSQL is triggered from the proxy route after a
successful `git-receive-pack`.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import shutil
import tempfile
import threading
from typing import Dict, Iterable, Optional

import httpx

# dulwich is an optional dependency (extra = "git-state"). CI installs don't
# include it, so imports must be lazy/guarded.
try:
    from dulwich.repo import Repo
    from dulwich.server import Backend
    from dulwich.web import HTTPGitApplication, make_server

    _DULWICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    Repo = None  # type: ignore[assignment]

    class Backend:  # type: ignore[no-redef]
        pass

    HTTPGitApplication = None  # type: ignore[assignment]
    make_server = None  # type: ignore[assignment]
    _DULWICH_AVAILABLE = False

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from letta.log import get_logger
from letta.server.rest_api.dependencies import HeaderParams, get_headers, get_letta_server

logger = get_logger(__name__)

# Routes are proxied to dulwich running on a separate port.
router = APIRouter(prefix="/git", tags=["git"], include_in_schema=False)

# Global storage for the server instance (set during app startup)
_server_instance = None

# org_id/agent_id -> temp working tree path (repo root, with .git inside)
_repo_cache: Dict[str, str] = {}
_repo_locks: Dict[str, threading.Lock] = {}


def _dulwich_repo_path_marker_file(cache_key: str) -> str:
    """Path to a marker file that stores the dulwich temp repo path.

    Dulwich runs in-process and mutates a repo materialized into a temp directory.
    We then need to locate that same temp directory after the push to persist the
    updated `.git/` contents back to object storage.

    In production we may have multiple FastAPI workers; in-memory `_repo_cache`
    is not shared across workers, so we store the repo_path in a small file under
    /tmp as a best-effort handoff. (Longer-term, we'll likely move dulwich to its
    own service/process and remove this.)
    """

    safe = cache_key.replace("/", "__")
    base = os.path.join(tempfile.gettempdir(), "letta-git-http")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"dulwich_repo_path__{safe}.txt")


# org_id for the currently-handled dulwich request (set by a WSGI wrapper).
_current_org_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("letta_git_http_org_id", default=None)


# Dulwich server globals
_dulwich_server = None
_dulwich_thread: Optional[threading.Thread] = None


def set_server_instance(server) -> None:
    """Set the Letta server instance for git operations. Called during app startup."""

    global _server_instance
    _server_instance = server


def _get_dulwich_port() -> int:
    return int(os.getenv("LETTA_GIT_HTTP_DULWICH_PORT", "8284"))


def start_dulwich_server(host: str = "127.0.0.1", port: Optional[int] = None) -> None:
    """Start a local dulwich HTTP server in a background thread.

    This is safe to call multiple times; only the first successful call will
    start a server in the current process.
    """

    global _dulwich_server, _dulwich_thread

    if not _DULWICH_AVAILABLE:
        logger.info("dulwich not installed; git smart HTTP is disabled")
        return

    if _dulwich_thread and _dulwich_thread.is_alive():
        return

    if port is None:
        port = _get_dulwich_port()

    # Ensure backend can access storage through the running server.
    if _server_instance is None:
        raise RuntimeError("Server instance not set (did you call set_server_instance?)")

    try:
        _dulwich_server = make_server(host, port, _git_wsgi_app)
    except OSError as e:
        # When running with multiple uvicorn workers, only one process can bind
        # to the configured port.
        logger.warning("Failed to bind dulwich git server on %s:%s: %s", host, port, e)
        return

    def _run():
        logger.info("Starting dulwich git HTTP server on http://%s:%s", host, port)
        try:
            _dulwich_server.serve_forever()
        except Exception:
            logger.exception("Dulwich git HTTP server crashed")

    _dulwich_thread = threading.Thread(target=_run, name="dulwich-git-http", daemon=True)
    _dulwich_thread.start()


def stop_dulwich_server() -> None:
    """Stop the local dulwich server (best-effort)."""

    global _dulwich_server
    if _dulwich_server is None:
        return
    try:
        _dulwich_server.shutdown()
    except Exception:
        logger.exception("Failed to shutdown dulwich server")


def _require_current_org_id() -> str:
    """Read the org_id set by the WSGI wrapper for the current request."""

    org_id = _current_org_id.get()
    if not org_id:
        raise RuntimeError("Missing org_id for git HTTP request")
    return org_id


def _resolve_org_id_from_wsgi_environ(environ: dict) -> Optional[str]:
    """Resolve org_id for dulwich, preferring X-Organization-Id.

    This is used by the dulwich WSGI wrapper. If X-Organization-Id is missing,
    we fall back to resolving via the authenticated user_id header.

    Note: dulwich is served on 127.0.0.1, so these headers should only be set by
    our trusted in-pod proxy layer.
    """

    org_id = environ.get("HTTP_X_ORGANIZATION_ID")
    if org_id:
        return org_id

    user_id = environ.get("HTTP_USER_ID")
    if not user_id:
        return None

    if _server_instance is None:
        return None

    try:
        # We are in a dulwich WSGI thread; run async DB lookup in a fresh loop.
        actor = asyncio.run(_server_instance.user_manager.get_actor_by_id_async(user_id))
        resolved = actor.organization_id
    except Exception:
        logger.exception("Failed to resolve org_id from user_id for dulwich request (user_id=%s)", user_id)
        return None

    return resolved


class GCSBackend(Backend):
    """Dulwich backend that materializes repos from GCS."""

    def open_repository(self, path: str | bytes):
        """Open a repository by path.

        dulwich passes paths like:
            /{agent_id}/state.git
            /{agent_id}/state.git/info/refs
            /{agent_id}/state.git/git-upload-pack
            /{agent_id}/state.git/git-receive-pack

        We map those to an on-disk repo cached in a temp dir.
        """

        if not _DULWICH_AVAILABLE or Repo is None:
            raise RuntimeError("dulwich not installed")

        if isinstance(path, (bytes, bytearray)):
            path = path.decode("utf-8", errors="surrogateescape")

        parts = path.strip("/").split("/")

        # Supported path form: /{agent_id}/state.git[/...]
        if "state.git" not in parts:
            raise ValueError(f"Invalid repository path (missing state.git): {path}")

        repo_idx = parts.index("state.git")
        if repo_idx != 1:
            raise ValueError(f"Invalid repository path (expected /{{agent_id}}/state.git): {path}")

        agent_id = parts[0]
        org_id = _require_current_org_id()

        cache_key = f"{org_id}/{agent_id}"
        logger.info("GCSBackend.open_repository: org=%s agent=%s", org_id, agent_id)

        lock = _repo_locks.setdefault(cache_key, threading.Lock())
        with lock:
            # Always refresh from GCS to avoid serving stale refs/objects when the
            # repo is mutated through non-git code paths (e.g. git-state APIs)
            # or when multiple app workers are running.
            old_repo_path = _repo_cache.pop(cache_key, None)
            if old_repo_path:
                shutil.rmtree(os.path.dirname(old_repo_path), ignore_errors=True)
                try:
                    os.unlink(_dulwich_repo_path_marker_file(cache_key))
                except FileNotFoundError:
                    pass

            repo_path = self._download_repo_sync(agent_id=agent_id, org_id=org_id)
            _repo_cache[cache_key] = repo_path

            # Persist repo_path for cross-worker post-push sync.
            try:
                with open(_dulwich_repo_path_marker_file(cache_key), "w") as f:
                    f.write(repo_path)
            except Exception:
                logger.exception("Failed to write repo_path marker for %s", cache_key)

            repo = Repo(repo_path)
            _prune_broken_refs(repo)
            return repo

    def _download_repo_sync(self, agent_id: str, org_id: str) -> str:
        """Synchronously download a repo from GCS.

        dulwich runs in a background thread (wsgiref server thread), so we should
        not assume we're on the main event loop.
        """

        if _server_instance is None:
            raise RuntimeError("Server instance not set (did you call set_server_instance?)")

        # This runs in a dulwich-managed WSGI thread, not an AnyIO worker thread.
        # Use a dedicated event loop to run the async download.
        return asyncio.run(self._download_repo(agent_id, org_id))

    async def _download_repo(self, agent_id: str, org_id: str) -> str:
        """Download repo from GCS into a temporary working tree."""

        storage = _server_instance.memory_repo_manager.git.storage
        storage_prefix = f"{org_id}/{agent_id}/repo.git"

        files = await storage.list_files(storage_prefix)
        if not files:
            # Create an empty repo on-demand so clients can `git clone` immediately.
            logger.info("Repository not found for agent %s; creating empty repo", agent_id)
            await _server_instance.memory_repo_manager.git.create_repo(
                agent_id=agent_id,
                org_id=org_id,
                initial_files={},
                author_name="Letta System",
                author_email="system@letta.ai",
            )
            files = await storage.list_files(storage_prefix)
            if not files:
                raise FileNotFoundError(f"Repository not found for agent {agent_id}")

        temp_dir = tempfile.mkdtemp(prefix="letta-git-http-")
        repo_path = os.path.join(temp_dir, "repo")
        git_dir = os.path.join(repo_path, ".git")
        os.makedirs(git_dir)

        # Ensure required git directories exist for fetch/push even if GCS doesn't
        # have any objects packed yet.
        for subdir in [
            "objects",
            os.path.join("objects", "pack"),
            os.path.join("objects", "info"),
            "refs",
            os.path.join("refs", "heads"),
            os.path.join("refs", "tags"),
            "info",
        ]:
            os.makedirs(os.path.join(git_dir, subdir), exist_ok=True)

        async def download_file(file_path: str):
            if file_path.startswith(storage_prefix):
                rel_path = file_path[len(storage_prefix) + 1 :]
            else:
                rel_path = file_path.split("/")[-1]

            if not rel_path:
                return

            local_path = os.path.join(git_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            content = await storage.download_bytes(file_path)
            with open(local_path, "wb") as f:
                f.write(content)

        await asyncio.gather(*[download_file(f) for f in files])
        logger.info("Downloaded %s files from GCS for agent %s", len(files), agent_id)

        return repo_path


def _prune_broken_refs(repo: Repo) -> int:
    """Remove refs that point at missing objects.

    This can happen if a prior push partially failed after updating refs but
    before all objects were persisted to backing storage.

    We prune these so dulwich doesn't advertise/resolve against corrupt refs,
    which can lead to `UnresolvedDeltas` during subsequent pushes.
    """

    removed = 0
    try:
        ref_names = list(repo.refs.keys())
    except Exception:
        logger.exception("Failed to enumerate refs for pruning")
        return 0

    for name in ref_names:
        # HEAD is commonly symbolic; skip.
        if name in {b"HEAD", "HEAD"}:
            continue
        try:
            sha = repo.refs[name]
        except Exception:
            continue
        if not sha:
            continue
        try:
            if sha not in repo.object_store:
                logger.warning("Pruning broken ref %r -> %r", name, sha)
                try:
                    repo.refs.remove_if_equals(name, sha)
                except Exception:
                    # Best-effort fallback
                    try:
                        del repo.refs[name]
                    except Exception:
                        pass
                removed += 1
        except Exception:
            logger.exception("Failed while checking ref %r", name)

    return removed


async def _sync_after_push(actor_id: str, agent_id: str) -> None:
    """Sync repo back to GCS and PostgreSQL after a successful push."""

    if _server_instance is None:
        logger.warning("Server instance not set; cannot sync after push")
        return

    try:
        actor = await _server_instance.user_manager.get_actor_by_id_async(actor_id)
    except Exception:
        logger.exception("Failed to resolve actor for post-push sync (actor_id=%s)", actor_id)
        return

    org_id = actor.organization_id
    cache_key = f"{org_id}/{agent_id}"

    repo_path = _repo_cache.get(cache_key)
    if not repo_path:
        # Cross-worker fallback: read marker file written by the dulwich process.
        try:
            with open(_dulwich_repo_path_marker_file(cache_key), "r") as f:
                repo_path = f.read().strip() or None
        except FileNotFoundError:
            repo_path = None

    if not repo_path:
        logger.warning("No cached repo for %s after push", cache_key)
        return

    if not os.path.exists(repo_path):
        logger.warning("Repo path %s does not exist after push", repo_path)
        return

    logger.info("Syncing repo after push: org=%s agent=%s", org_id, agent_id)

    storage = _server_instance.memory_repo_manager.git.storage
    storage_prefix = f"{org_id}/{agent_id}/repo.git"
    git_dir = os.path.join(repo_path, ".git")

    upload_tasks = []
    for root, _dirs, files in os.walk(git_dir):
        for filename in files:
            local_file = os.path.join(root, filename)
            rel_path = os.path.relpath(local_file, git_dir)
            storage_path = f"{storage_prefix}/{rel_path}"

            with open(local_file, "rb") as f:
                content = f.read()

            upload_tasks.append(storage.upload_bytes(storage_path, content))

    await asyncio.gather(*upload_tasks)
    logger.info("Uploaded %s files to GCS", len(upload_tasks))

    # Sync blocks to Postgres (if using GitEnabledBlockManager).
    #
    # Keep the same pattern as API-driven edits: read from the source of truth
    # in object storage after persisting the pushed refs/objects, rather than
    # relying on a working tree checkout under repo_path/.
    from letta.services.block_manager_git import GitEnabledBlockManager

    if isinstance(_server_instance.block_manager, GitEnabledBlockManager):
        try:
            files = await _server_instance.memory_repo_manager.git.get_files(
                agent_id=agent_id,
                org_id=org_id,
                ref="HEAD",
            )
        except Exception:
            logger.exception("Failed to read repo files from storage for post-push block sync (agent=%s)", agent_id)
            files = {}

        expected_labels = set()
        synced = 0
        for file_path, content in files.items():
            if not file_path.startswith("memory/") or not file_path.endswith(".md"):
                continue

            label = file_path[len("memory/") : -3]
            expected_labels.add(label)
            try:
                await _server_instance.block_manager._sync_block_to_postgres(
                    agent_id=agent_id,
                    label=label,
                    value=content,
                    actor=actor,
                )
                synced += 1
                logger.info("Synced block %s to PostgreSQL", label)
            except Exception:
                logger.exception("Failed to sync block %s to PostgreSQL (agent=%s)", label, agent_id)

        if synced == 0:
            logger.warning("No memory/*.md files found in repo HEAD during post-push sync (agent=%s)", agent_id)
        else:
            # Detach blocks that were removed in git.
            #
            # We treat git as the source of truth for which blocks are attached to
            # this agent. If a blocks/*.md file disappears from HEAD, detach the
            # corresponding block from the agent in Postgres.
            try:
                existing_blocks = await _server_instance.agent_manager.list_agent_blocks_async(
                    agent_id=agent_id,
                    actor=actor,
                    before=None,
                    after=None,
                    limit=1000,
                    ascending=True,
                )
                existing_by_label = {b.label: b for b in existing_blocks}
                removed_labels = set(existing_by_label.keys()) - expected_labels

                for label in sorted(removed_labels):
                    block = existing_by_label.get(label)
                    if not block:
                        continue
                    await _server_instance.agent_manager.detach_block_async(
                        agent_id=agent_id,
                        block_id=block.id,
                        actor=actor,
                    )
                    logger.info("Detached block %s from agent (removed from git)", label)
            except Exception:
                logger.exception("Failed detaching removed blocks during post-push sync (agent=%s)", agent_id)

    # Cleanup local cache
    _repo_cache.pop(cache_key, None)
    try:
        os.unlink(_dulwich_repo_path_marker_file(cache_key))
    except FileNotFoundError:
        pass
    shutil.rmtree(os.path.dirname(repo_path), ignore_errors=True)


def _parse_agent_id_from_repo_path(path: str) -> Optional[str]:
    """Extract agent_id from a git HTTP path.

    Expected path form:
      - {agent_id}/state.git/...
    """

    parts = path.strip("/").split("/")
    if len(parts) < 2:
        return None

    if parts[1] != "state.git":
        return None

    return parts[0]


def _filter_out_hop_by_hop_headers(headers: Iterable[tuple[str, str]]) -> Dict[str, str]:
    # RFC 7230 hop-by-hop headers that should not be forwarded
    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }

    out: Dict[str, str] = {}
    for k, v in headers:
        lk = k.lower()
        if lk in hop_by_hop:
            continue
        out[k] = v
    return out


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])  # pragma: no cover
async def proxy_git_http(
    path: str,
    request: Request,
    server=Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """Proxy `/v1/git/*` requests to the local dulwich WSGI server."""

    if not _DULWICH_AVAILABLE:
        return JSONResponse(
            status_code=501,
            content={
                "detail": "git smart HTTP is disabled (dulwich not installed)",
            },
        )

    # Ensure server is running (best-effort). We also start it during lifespan.
    start_dulwich_server()

    port = _get_dulwich_port()
    url = f"http://127.0.0.1:{port}/{path}"

    req_headers = _filter_out_hop_by_hop_headers(request.headers.items())
    # Avoid sending FastAPI host/length; httpx will compute
    req_headers.pop("host", None)
    req_headers.pop("content-length", None)

    # Resolve org_id from the authenticated actor + agent and forward to dulwich.
    agent_id = _parse_agent_id_from_repo_path(path)
    if agent_id is not None:
        actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
        # Authorization check: ensure the actor can access this agent.
        await server.agent_manager.get_agent_by_id_async(agent_id=agent_id, actor=actor, include_relationships=[])

        # Ensure we set exactly one X-Organization-Id header (avoid duplicate casing).
        for k in list(req_headers.keys()):
            if k.lower() == "x-organization-id":
                req_headers.pop(k, None)
        # Use the authenticated actor's org; AgentState may not carry an organization field.
        req_headers["X-Organization-Id"] = actor.organization_id

    logger.info(
        "proxy_git_http: method=%s path=%s parsed_agent_id=%s actor_id=%s has_user_id_hdr=%s x_org_hdr=%s",
        request.method,
        path,
        agent_id,
        headers.actor_id,
        bool(request.headers.get("user_id")),
        req_headers.get("X-Organization-Id") or req_headers.get("x-organization-id"),
    )

    async def _body_iter():
        async for chunk in request.stream():
            yield chunk

    client = httpx.AsyncClient(timeout=None)
    req = client.build_request(
        method=request.method,
        url=url,
        params=request.query_params,
        headers=req_headers,
        content=_body_iter() if request.method not in {"GET", "HEAD"} else None,
    )
    upstream = await client.send(req, stream=True)

    resp_headers = _filter_out_hop_by_hop_headers(upstream.headers.items())

    # If this was a push, trigger our sync.
    if request.method == "POST" and path.endswith("git-receive-pack") and upstream.status_code < 400:
        agent_id = _parse_agent_id_from_repo_path(path)
        if agent_id is not None:
            try:
                actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
                # Authorization check: ensure the actor can access this agent.
                await server.agent_manager.get_agent_by_id_async(agent_id=agent_id, actor=actor, include_relationships=[])
                # Fire-and-forget; do not block git client response.
                asyncio.create_task(_sync_after_push(actor.id, agent_id))
            except Exception:
                logger.exception("Failed to trigger post-push sync (agent_id=%s)", agent_id)

    async def _aclose_upstream_and_client() -> None:
        try:
            await upstream.aclose()
        finally:
            await client.aclose()

    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
        background=BackgroundTask(_aclose_upstream_and_client),
    )


def _org_header_middleware(app):
    """WSGI wrapper to capture org_id from proxied requests.

    FastAPI proxies requests to the dulwich server and injects `X-Organization-Id`.
    Dulwich itself only passes repository *paths* into the Backend, so we capture
    the org_id from the WSGI environ and stash it in a contextvar.

    Important: WSGI apps can return iterables/generators, and the server may
    iterate the response body *after* this wrapper returns. We must therefore
    keep the contextvar set for the duration of iteration.

    Defensive fallback: if X-Organization-Id is missing, attempt to derive org_id
    from `user_id` (set by our auth proxy layer).
    """

    def _wrapped(environ, start_response):
        org_id = _resolve_org_id_from_wsgi_environ(environ)

        logger.info(
            "dulwich_wsgi: path=%s remote=%s has_x_org=%s has_user_id=%s resolved_org=%s",
            environ.get("PATH_INFO"),
            environ.get("REMOTE_ADDR"),
            bool(environ.get("HTTP_X_ORGANIZATION_ID")),
            bool(environ.get("HTTP_USER_ID")),
            org_id,
        )

        token = _current_org_id.set(org_id)

        try:
            app_iter = app(environ, start_response)
        except Exception:
            _current_org_id.reset(token)
            raise

        def _iter():
            try:
                yield from app_iter
            finally:
                try:
                    if hasattr(app_iter, "close"):
                        app_iter.close()
                finally:
                    _current_org_id.reset(token)

        return _iter()

    return _wrapped


# dulwich WSGI app (optional)
_backend = GCSBackend()
_git_wsgi_app = _org_header_middleware(HTTPGitApplication(_backend)) if _DULWICH_AVAILABLE and HTTPGitApplication is not None else None
