"""Git operations for memory repositories using dulwich.

Dulwich is a pure-Python implementation of Git that allows us to
manipulate git repositories without requiring libgit2 or the git CLI.

This module provides high-level operations for working with git repos
stored in object storage (GCS/S3).
"""

import asyncio
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from letta.data_sources.redis_client import get_redis_client
from letta.log import get_logger
from letta.schemas.memory_repo import FileChange, MemoryCommit
from letta.services.memory_repo.storage.base import StorageBackend

logger = get_logger(__name__)


class GitOperations:
    """High-level git operations for memory repositories.

    This class provides git operations that work with repositories
    stored in object storage. It downloads the repo to a temp directory,
    performs operations, and uploads the changes back.

    For efficiency with small repos (100s of files), we use a full
    checkout model. For larger repos, we could optimize to work with
    packfiles directly.

    Requirements:
        pip install dulwich
    """

    def __init__(self, storage: StorageBackend):
        """Initialize git operations.

        Args:
            storage: Storage backend for repo persistence
        """
        self.storage = storage
        self._dulwich = None

    def _get_dulwich(self):
        """Lazily import dulwich."""
        if self._dulwich is None:
            try:
                import dulwich
                import dulwich.objects
                import dulwich.porcelain
                import dulwich.repo

                self._dulwich = dulwich
            except ImportError:
                raise ImportError("dulwich is required for git operations. Install with: pip install dulwich")
        return self._dulwich

    def _repo_path(self, agent_id: str, org_id: str) -> str:
        """Get the storage path for an agent's repo."""
        return f"{org_id}/{agent_id}/repo.git"

    async def create_repo(
        self,
        agent_id: str,
        org_id: str,
        initial_files: Optional[Dict[str, str]] = None,
        author_name: str = "Letta System",
        author_email: str = "system@letta.ai",
    ) -> str:
        """Create a new git repository for an agent.

        Args:
            agent_id: Agent ID
            org_id: Organization ID
            initial_files: Optional initial files to commit
            author_name: Author name for initial commit
            author_email: Author email for initial commit

        Returns:
            Initial commit SHA
        """
        dulwich = self._get_dulwich()

        def _create():
            # Create a temporary directory for the repo
            temp_dir = tempfile.mkdtemp(prefix="letta-memrepo-")
            try:
                repo_path = os.path.join(temp_dir, "repo")
                os.makedirs(repo_path)

                # Initialize a new repository
                dulwich.repo.Repo.init(repo_path)

                # Use `main` as the default branch (git's modern default).
                head_path = os.path.join(repo_path, ".git", "HEAD")
                with open(head_path, "wb") as f:
                    f.write(b"ref: refs/heads/main\n")

                # Add initial files if provided
                if initial_files:
                    for file_path, content in initial_files.items():
                        full_path = os.path.join(repo_path, file_path)
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        with open(full_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        # Stage the file
                        dulwich.porcelain.add(repo_path, paths=[file_path])
                else:
                    # Create an empty .letta directory to initialize
                    letta_dir = os.path.join(repo_path, ".letta")
                    os.makedirs(letta_dir, exist_ok=True)
                    config_path = os.path.join(letta_dir, "config.json")
                    with open(config_path, "w") as f:
                        f.write('{"version": 1}')
                    dulwich.porcelain.add(repo_path, paths=[".letta/config.json"])

                # Create initial commit using porcelain (dulwich 1.0+ API)
                commit_sha = dulwich.porcelain.commit(
                    repo_path,
                    message=b"Initial commit",
                    committer=f"{author_name} <{author_email}>".encode(),
                    author=f"{author_name} <{author_email}>".encode(),
                )

                # Return the repo directory and commit SHA for upload
                return repo_path, commit_sha.decode() if isinstance(commit_sha, bytes) else str(commit_sha)
            except Exception:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise

        repo_path, commit_sha = await asyncio.to_thread(_create)

        try:
            # Upload the repo to storage
            await self._upload_repo(repo_path, agent_id, org_id)
            return commit_sha
        finally:
            # Clean up temp directory
            shutil.rmtree(os.path.dirname(repo_path), ignore_errors=True)

    async def _upload_repo(self, local_repo_path: str, agent_id: str, org_id: str) -> None:
        """Upload a local repo to storage (full upload)."""
        t_start = time.perf_counter()
        storage_prefix = self._repo_path(agent_id, org_id)

        # Walk through the .git directory and collect all files
        git_dir = os.path.join(local_repo_path, ".git")
        upload_tasks = []
        total_bytes = 0

        t0 = time.perf_counter()
        for root, dirs, files in os.walk(git_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                rel_path = os.path.relpath(local_path, git_dir)
                storage_path = f"{storage_prefix}/{rel_path}"

                with open(local_path, "rb") as f:
                    content = f.read()

                total_bytes += len(content)
                upload_tasks.append((storage_path, content))
        read_time = (time.perf_counter() - t0) * 1000
        logger.info(f"[GIT_PERF] _upload_repo read files took {read_time:.2f}ms files={len(upload_tasks)}")

        # Upload all files in parallel
        t0 = time.perf_counter()
        await asyncio.gather(*[self.storage.upload_bytes(path, content) for path, content in upload_tasks])
        upload_time = (time.perf_counter() - t0) * 1000

        total_time = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[GIT_PERF] _upload_repo TOTAL {total_time:.2f}ms "
            f"files={len(upload_tasks)} bytes={total_bytes} "
            f"upload_time={upload_time:.2f}ms"
        )

    @staticmethod
    def _snapshot_git_files(git_dir: str) -> Dict[str, float]:
        """Snapshot mtime of all files under .git/ for delta detection."""
        snapshot = {}
        for root, _dirs, files in os.walk(git_dir):
            for filename in files:
                path = os.path.join(root, filename)
                snapshot[path] = os.path.getmtime(path)
        return snapshot

    async def _upload_delta(
        self,
        local_repo_path: str,
        agent_id: str,
        org_id: str,
        before_snapshot: Dict[str, float],
    ) -> None:
        """Upload only new/modified files since before_snapshot."""
        t_start = time.perf_counter()
        storage_prefix = self._repo_path(agent_id, org_id)
        git_dir = os.path.join(local_repo_path, ".git")

        upload_tasks = []
        total_bytes = 0

        for root, _dirs, files in os.walk(git_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                old_mtime = before_snapshot.get(local_path)
                # New file or modified since snapshot
                if old_mtime is None or os.path.getmtime(local_path) != old_mtime:
                    rel_path = os.path.relpath(local_path, git_dir)
                    storage_path = f"{storage_prefix}/{rel_path}"
                    with open(local_path, "rb") as f:
                        content = f.read()
                    total_bytes += len(content)
                    upload_tasks.append((storage_path, content))

        t0 = time.perf_counter()
        await asyncio.gather(*[self.storage.upload_bytes(path, content) for path, content in upload_tasks])
        upload_time = (time.perf_counter() - t0) * 1000

        total_time = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[GIT_PERF] _upload_delta TOTAL {total_time:.2f}ms "
            f"files={len(upload_tasks)} bytes={total_bytes} "
            f"upload_time={upload_time:.2f}ms"
        )

    async def _download_repo(self, agent_id: str, org_id: str) -> str:
        """Download a repo from storage to a temp directory.

        Returns:
            Path to the temporary repo directory
        """
        t_start = time.perf_counter()
        storage_prefix = self._repo_path(agent_id, org_id)

        # List all files in the repo
        t0 = time.perf_counter()
        files = await self.storage.list_files(storage_prefix)
        list_time = (time.perf_counter() - t0) * 1000
        logger.info(f"[GIT_PERF] _download_repo storage.list_files took {list_time:.2f}ms files_count={len(files)}")

        if not files:
            raise FileNotFoundError(f"No repository found for agent {agent_id}")

        # Create temp directory
        t0 = time.perf_counter()
        temp_dir = tempfile.mkdtemp(prefix="letta-memrepo-")
        repo_path = os.path.join(temp_dir, "repo")
        git_dir = os.path.join(repo_path, ".git")
        os.makedirs(git_dir)
        mkdir_time = (time.perf_counter() - t0) * 1000
        logger.info(f"[GIT_PERF] _download_repo tempdir creation took {mkdir_time:.2f}ms path={temp_dir}")

        # Compute local paths and create directories first
        file_info = []
        for file_path in files:
            if file_path.startswith(storage_prefix):
                rel_path = file_path[len(storage_prefix) + 1 :]
            else:
                rel_path = file_path.split("/")[-1] if "/" in file_path else file_path

            local_path = os.path.join(git_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            file_info.append((file_path, local_path))

        # Download all files in parallel
        t0 = time.perf_counter()
        download_tasks = [self.storage.download_bytes(fp) for fp, _ in file_info]
        contents = await asyncio.gather(*download_tasks)
        download_time = (time.perf_counter() - t0) * 1000
        total_bytes = sum(len(c) for c in contents)
        logger.info(f"[GIT_PERF] _download_repo parallel download took {download_time:.2f}ms files={len(files)} bytes={total_bytes}")

        # Write all files to disk
        t0 = time.perf_counter()
        for (_, local_path), content in zip(file_info, contents):
            with open(local_path, "wb") as f:
                f.write(content)
        write_time = (time.perf_counter() - t0) * 1000

        total_time = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[GIT_PERF] _download_repo TOTAL {total_time:.2f}ms "
            f"files={len(files)} bytes={total_bytes} "
            f"download_time={download_time:.2f}ms write_time={write_time:.2f}ms"
        )

        return repo_path

    async def get_files(
        self,
        agent_id: str,
        org_id: str,
        ref: str = "HEAD",
    ) -> Dict[str, str]:
        """Get all files at a specific ref.

        Args:
            agent_id: Agent ID
            org_id: Organization ID
            ref: Git ref (commit SHA, branch name, or 'HEAD')

        Returns:
            Dict mapping file paths to content
        """
        dulwich = self._get_dulwich()
        repo_path = await self._download_repo(agent_id, org_id)

        try:

            def _get_files():
                repo = dulwich.repo.Repo(repo_path)

                # Resolve ref to commit
                if ref == "HEAD":
                    commit_sha = repo.head()
                else:
                    # Try as branch name first
                    try:
                        commit_sha = repo.refs[f"refs/heads/{ref}".encode()]
                    except KeyError:
                        # Try as commit SHA
                        commit_sha = ref.encode() if isinstance(ref, str) else ref

                commit = repo[commit_sha]
                tree = repo[commit.tree]

                # Walk the tree and get all files
                files = {}
                self._walk_tree(repo, tree, "", files)
                return files

            return await asyncio.to_thread(_get_files)
        finally:
            shutil.rmtree(os.path.dirname(repo_path), ignore_errors=True)

    def _walk_tree(self, repo, tree, prefix: str, files: Dict[str, str]) -> None:
        """Recursively walk a git tree and collect files."""
        dulwich = self._get_dulwich()
        for entry in tree.items():
            name = entry.path.decode() if isinstance(entry.path, bytes) else entry.path
            path = f"{prefix}/{name}" if prefix else name
            obj = repo[entry.sha]

            if isinstance(obj, dulwich.objects.Blob):
                try:
                    files[path] = obj.data.decode("utf-8")
                except UnicodeDecodeError:
                    # Skip binary files
                    pass
            elif isinstance(obj, dulwich.objects.Tree):
                self._walk_tree(repo, obj, path, files)

    async def commit(
        self,
        agent_id: str,
        org_id: str,
        changes: List[FileChange],
        message: str,
        author_name: str = "Letta Agent",
        author_email: str = "agent@letta.ai",
        branch: str = "main",
    ) -> MemoryCommit:
        """Commit changes to the repository.

        Uses a Redis lock to prevent concurrent modifications.

        Args:
            agent_id: Agent ID
            org_id: Organization ID
            changes: List of file changes
            message: Commit message
            author_name: Author name
            author_email: Author email
            branch: Branch to commit to

        Returns:
            MemoryCommit with commit details

        Raises:
            MemoryRepoBusyError: If another operation is in progress
        """
        t_start = time.perf_counter()
        logger.info(f"[GIT_PERF] GitOperations.commit START agent={agent_id} changes={len(changes)}")

        # Acquire lock to prevent concurrent modifications
        t0 = time.perf_counter()
        redis_client = await get_redis_client()
        lock_token = f"commit:{uuid.uuid4().hex}"
        lock = await redis_client.acquire_memory_repo_lock(agent_id, lock_token)
        logger.info(f"[GIT_PERF] acquire_memory_repo_lock took {(time.perf_counter() - t0) * 1000:.2f}ms")

        try:
            t0 = time.perf_counter()
            result = await self._commit_with_lock(
                agent_id=agent_id,
                org_id=org_id,
                changes=changes,
                message=message,
                author_name=author_name,
                author_email=author_email,
                branch=branch,
            )
            logger.info(f"[GIT_PERF] _commit_with_lock took {(time.perf_counter() - t0) * 1000:.2f}ms")

            total_time = (time.perf_counter() - t_start) * 1000
            logger.info(f"[GIT_PERF] GitOperations.commit TOTAL {total_time:.2f}ms")
            return result
        finally:
            # Release lock
            t0 = time.perf_counter()
            if lock:
                try:
                    await lock.release()
                except Exception as e:
                    logger.warning(f"Failed to release lock for agent {agent_id}: {e}")
                    await redis_client.release_memory_repo_lock(agent_id)
            logger.info(f"[GIT_PERF] lock release took {(time.perf_counter() - t0) * 1000:.2f}ms")

    async def _commit_with_lock(
        self,
        agent_id: str,
        org_id: str,
        changes: List[FileChange],
        message: str,
        author_name: str = "Letta Agent",
        author_email: str = "agent@letta.ai",
        branch: str = "main",
    ) -> MemoryCommit:
        """Internal commit implementation (called while holding lock)."""
        t_start = time.perf_counter()
        dulwich = self._get_dulwich()

        # Download repo from GCS to temp dir
        t0 = time.perf_counter()
        repo_path = await self._download_repo(agent_id, org_id)
        download_time = (time.perf_counter() - t0) * 1000
        logger.info(f"[GIT_PERF] _commit_with_lock download phase took {download_time:.2f}ms")

        try:
            # Snapshot git objects before commit for delta upload
            git_dir = os.path.join(repo_path, ".git")
            before_snapshot = self._snapshot_git_files(git_dir)

            def _commit():
                t_git_start = time.perf_counter()
                repo = dulwich.repo.Repo(repo_path)

                # Checkout the working directory
                t0_reset = time.perf_counter()
                dulwich.porcelain.reset(repo, "hard")
                reset_time = (time.perf_counter() - t0_reset) * 1000

                # Apply changes
                files_changed = []
                additions = 0
                deletions = 0
                apply_time = 0

                for change in changes:
                    t0_apply = time.perf_counter()
                    file_path = change.path.lstrip("/")
                    full_path = os.path.join(repo_path, file_path)

                    if change.change_type == "delete" or change.content is None:
                        # Delete file
                        if os.path.exists(full_path):
                            with open(full_path, "r") as f:
                                deletions += len(f.read())
                            os.remove(full_path)
                            dulwich.porcelain.remove(repo_path, paths=[file_path])
                    else:
                        # Add or modify file
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)

                        # Calculate additions/deletions
                        if os.path.exists(full_path):
                            with open(full_path, "r") as f:
                                old_content = f.read()
                            deletions += len(old_content)
                        additions += len(change.content)

                        with open(full_path, "w", encoding="utf-8") as f:
                            f.write(change.content)
                        dulwich.porcelain.add(repo_path, paths=[file_path])

                    files_changed.append(file_path)
                    apply_time += (time.perf_counter() - t0_apply) * 1000

                # Get parent SHA
                try:
                    parent_sha = repo.head().decode()
                except Exception:
                    parent_sha = None

                # Create commit using porcelain (dulwich 1.0+ API)
                t0_commit = time.perf_counter()
                commit_sha = dulwich.porcelain.commit(
                    repo_path,
                    message=message.encode(),
                    committer=f"{author_name} <{author_email}>".encode(),
                    author=f"{author_name} <{author_email}>".encode(),
                )
                commit_time = (time.perf_counter() - t0_commit) * 1000

                sha_str = commit_sha.decode() if isinstance(commit_sha, bytes) else str(commit_sha)

                git_total = (time.perf_counter() - t_git_start) * 1000
                logger.info(
                    f"[GIT_PERF] _commit git operations: reset={reset_time:.2f}ms "
                    f"apply_changes={apply_time:.2f}ms commit={commit_time:.2f}ms total={git_total:.2f}ms"
                )

                return MemoryCommit(
                    sha=sha_str,
                    parent_sha=parent_sha,
                    message=message,
                    author_type="agent" if "agent" in author_email.lower() else "user",
                    author_id=agent_id,
                    author_name=author_name,
                    timestamp=datetime.now(timezone.utc),
                    files_changed=files_changed,
                    additions=additions,
                    deletions=deletions,
                )

            t0 = time.perf_counter()
            commit = await asyncio.to_thread(_commit)
            git_thread_time = (time.perf_counter() - t0) * 1000
            logger.info(f"[GIT_PERF] _commit_with_lock git thread took {git_thread_time:.2f}ms")

            # Upload only new/modified objects (delta)
            t0 = time.perf_counter()
            await self._upload_delta(repo_path, agent_id, org_id, before_snapshot)
            upload_time = (time.perf_counter() - t0) * 1000
            logger.info(f"[GIT_PERF] _commit_with_lock upload phase (delta) took {upload_time:.2f}ms")

            total_time = (time.perf_counter() - t_start) * 1000
            logger.info(
                f"[GIT_PERF] _commit_with_lock TOTAL {total_time:.2f}ms "
                f"(download={download_time:.2f}ms git={git_thread_time:.2f}ms upload={upload_time:.2f}ms)"
            )

            return commit
        finally:
            t0 = time.perf_counter()
            shutil.rmtree(os.path.dirname(repo_path), ignore_errors=True)
            logger.info(f"[GIT_PERF] cleanup temp dir took {(time.perf_counter() - t0) * 1000:.2f}ms")

    async def get_history(
        self,
        agent_id: str,
        org_id: str,
        path: Optional[str] = None,
        limit: int = 50,
    ) -> List[MemoryCommit]:
        """Get commit history.

        Args:
            agent_id: Agent ID
            org_id: Organization ID
            path: Optional file path to filter by
            limit: Maximum number of commits to return

        Returns:
            List of commits, newest first
        """
        dulwich = self._get_dulwich()
        repo_path = await self._download_repo(agent_id, org_id)

        try:

            def _get_history():
                repo = dulwich.repo.Repo(repo_path)
                commits = []

                # Walk the commit history
                walker = repo.get_walker(max_entries=limit)

                for entry in walker:
                    commit = entry.commit
                    sha = commit.id.decode() if isinstance(commit.id, bytes) else str(commit.id)
                    parent_sha = commit.parents[0].decode() if commit.parents else None

                    # Parse author
                    author_str = commit.author.decode() if isinstance(commit.author, bytes) else commit.author
                    author_name = author_str.split("<")[0].strip() if "<" in author_str else author_str

                    commits.append(
                        MemoryCommit(
                            sha=sha,
                            parent_sha=parent_sha,
                            message=commit.message.decode() if isinstance(commit.message, bytes) else commit.message,
                            author_type="system",
                            author_id="",
                            author_name=author_name,
                            timestamp=datetime.fromtimestamp(commit.commit_time, tz=timezone.utc),
                            files_changed=[],  # Would need to compute diff for this
                            additions=0,
                            deletions=0,
                        )
                    )

                return commits

            return await asyncio.to_thread(_get_history)
        finally:
            shutil.rmtree(os.path.dirname(repo_path), ignore_errors=True)

    async def get_head_sha(self, agent_id: str, org_id: str) -> str:
        """Get the current HEAD commit SHA.

        Args:
            agent_id: Agent ID
            org_id: Organization ID

        Returns:
            HEAD commit SHA
        """
        dulwich = self._get_dulwich()
        repo_path = await self._download_repo(agent_id, org_id)

        try:

            def _get_head():
                repo = dulwich.repo.Repo(repo_path)
                head = repo.head()
                return head.decode() if isinstance(head, bytes) else str(head)

            return await asyncio.to_thread(_get_head)
        finally:
            shutil.rmtree(os.path.dirname(repo_path), ignore_errors=True)

    async def delete_repo(self, agent_id: str, org_id: str) -> None:
        """Delete an agent's repository from storage.

        Args:
            agent_id: Agent ID
            org_id: Organization ID
        """
        storage_prefix = self._repo_path(agent_id, org_id)
        await self.storage.delete_prefix(storage_prefix)
        logger.info(f"Deleted repository for agent {agent_id}")
