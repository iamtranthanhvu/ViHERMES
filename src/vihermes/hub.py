"""Move trained models between hosts through a private Hugging Face repo, pinned by commit hash.

Set ``VIHERMES_HUB_DIR`` to use a plain local directory instead of Hugging Face; offline dry
runs use it to exercise the same code path without network access or a token. The token is read
only from ``HF_TOKEN`` and is never written to a file, a log line or an error message.
"""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

COMMIT = re.compile(r"[0-9a-f]{40}")
HUB_FILE = "hub.json"


def _local_root():
    root = os.environ.get("VIHERMES_HUB_DIR")
    return Path(root) if root else None


def _token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN in the environment (a fine-grained token for the repo)")
    return token


def _folder(path):
    path = Path(path)
    if not path.is_dir() or not any(path.iterdir()):
        raise ValueError(f"Model folder {path} does not exist or is empty")
    return path


def _record_meta(meta_path, repo_id, revision):
    meta_path = Path(meta_path)
    data = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    data["hub"] = {"repo": repo_id, "revision": revision}
    meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _content_hash(path):
    digest = hashlib.sha1()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def publish(path, repo_id, meta_path=None):
    """Upload a folder to a private repo and return the commit hash it was stored under."""
    path = _folder(path)
    local = _local_root()
    if local is not None:
        revision = _content_hash(path)
        target = local / repo_id / revision
        if not target.exists():
            shutil.copytree(path, target)
    else:
        token = _token()
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
        if not api.repo_info(repo_id, repo_type="model").private:
            raise ValueError(f"{repo_id} is public; model repos must stay private")
        revision = api.upload_folder(
            folder_path=str(path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add trained weights",
        ).oid
    if meta_path:
        _record_meta(meta_path, repo_id, revision)
    return revision


def fetch(repo_id, revision, dest):
    """Download exactly the pinned revision into ``dest`` and check every file arrived."""
    if not COMMIT.fullmatch(revision or ""):
        raise ValueError("Pin the revision with a full 40-character lowercase commit hash")
    dest = Path(dest)
    local = _local_root()
    if local is not None:
        source = local / repo_id / revision
        if not source.is_dir():
            raise ValueError(f"{repo_id}@{revision} not found in {local}")
        shutil.copytree(source, dest, dirs_exist_ok=True)
        expected = [str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()]
    else:
        token = _token()
        from huggingface_hub import HfApi, snapshot_download

        snapshot_download(
            repo_id=repo_id, revision=revision, local_dir=str(dest), token=token, repo_type="model"
        )
        expected = HfApi(token=token).list_repo_files(repo_id, revision=revision, repo_type="model")
    missing = [name for name in expected if not (dest / name).is_file()]
    if missing:
        raise ValueError(f"Download incomplete; missing: {', '.join(sorted(missing))}")
    (dest / HUB_FILE).write_text(
        json.dumps({"repo": repo_id, "revision": revision}, indent=2) + "\n", encoding="utf-8"
    )
    return dest
