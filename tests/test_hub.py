import json
import os
import subprocess
import sys
import types

import pytest

from vihermes import hub

REV = "a" * 40


class FakeCommit:
    oid = REV


class FakeApi:
    """Stands in for huggingface_hub.HfApi; records what would have been sent."""

    repos = {}  # repo_id -> {"private": bool, "files": {name: bytes}}
    tokens = []

    def __init__(self, token=None):
        FakeApi.tokens.append(token)

    def create_repo(self, repo_id, private, exist_ok, repo_type):
        FakeApi.repos.setdefault(repo_id, {"private": private, "files": {}})

    def repo_info(self, repo_id, repo_type):
        return types.SimpleNamespace(private=FakeApi.repos[repo_id]["private"])

    def upload_folder(self, folder_path, repo_id, repo_type, commit_message):
        for name in os.listdir(folder_path):
            FakeApi.repos[repo_id]["files"][name] = open(
                os.path.join(folder_path, name), "rb"
            ).read()
        return FakeCommit()

    def list_repo_files(self, repo_id, revision, repo_type):
        assert revision == REV
        return list(FakeApi.repos[repo_id]["files"])


def fake_snapshot_download(repo_id, revision, local_dir, token=None, repo_type="model"):
    assert revision == REV
    os.makedirs(local_dir, exist_ok=True)
    for name, data in FakeApi.repos[repo_id]["files"].items():
        if name != "dropped.bin":
            with open(os.path.join(local_dir, name), "wb") as f:
                f.write(data)
    return local_dir


@pytest.fixture
def fake_hf(monkeypatch):
    FakeApi.repos, FakeApi.tokens = {}, []
    module = types.SimpleNamespace(HfApi=FakeApi, snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.delenv("VIHERMES_HUB_DIR", raising=False)
    return FakeApi


@pytest.fixture
def model_dir(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    (folder / "weights.bin").write_bytes(b"123")
    return folder


def test_publish_creates_private_repo_and_returns_commit_hash(fake_hf, model_dir, tmp_path):
    meta = tmp_path / "train_meta.json"
    meta.write_text(json.dumps({"seed": 42}))
    revision = hub.publish(model_dir, "user/model", meta_path=meta)
    assert revision == REV
    assert fake_hf.repos["user/model"]["private"] is True
    assert set(fake_hf.repos["user/model"]["files"]) == {"config.json", "weights.bin"}
    saved = json.loads(meta.read_text())
    assert saved["seed"] == 42 and saved["hub"] == {"repo": "user/model", "revision": REV}
    assert "hf_secret_value" not in meta.read_text()


def test_publish_refuses_a_public_repo(fake_hf, model_dir):
    fake_hf.repos["user/model"] = {"private": False, "files": {}}
    with pytest.raises(ValueError, match="public"):
        hub.publish(model_dir, "user/model")
    assert not fake_hf.repos["user/model"]["files"]


def test_publish_needs_a_token_and_never_reveals_it(fake_hf, model_dir, monkeypatch):
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(ValueError, match="HF_TOKEN"):
        hub.publish(model_dir, "user/model")


def test_publish_rejects_an_empty_or_missing_folder(fake_hf, tmp_path):
    with pytest.raises(ValueError, match="empty|exist"):
        hub.publish(tmp_path / "nope", "user/model")


def test_fetch_requires_a_pinned_commit_hash(fake_hf, tmp_path):
    for bad in ("main", "abc123", "A" * 40, ""):
        with pytest.raises(ValueError, match="commit hash"):
            hub.fetch("user/model", bad, tmp_path / "out")


def test_fetch_downloads_the_pinned_revision_and_records_it(fake_hf, model_dir, tmp_path):
    hub.publish(model_dir, "user/model")
    out = hub.fetch("user/model", REV, tmp_path / "out")
    assert (out / "weights.bin").read_bytes() == b"123"
    assert json.loads((out / "hub.json").read_text()) == {"repo": "user/model", "revision": REV}


def test_fetch_fails_when_a_file_is_missing(fake_hf, model_dir, tmp_path):
    hub.publish(model_dir, "user/model")
    fake_hf.repos["user/model"]["files"]["dropped.bin"] = b"x"
    with pytest.raises(ValueError, match="dropped.bin"):
        hub.fetch("user/model", REV, tmp_path / "out")


def test_local_hub_directory_roundtrip_for_offline_runs(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("VIHERMES_HUB_DIR", str(tmp_path / "hub"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    revision = hub.publish(model_dir, "user/model")
    assert len(revision) == 40 and revision == hub.publish(model_dir, "user/model")
    out = hub.fetch("user/model", revision, tmp_path / "out")
    assert (out / "weights.bin").read_bytes() == b"123"
    with pytest.raises(ValueError, match="not found"):
        hub.fetch("user/model", "b" * 40, tmp_path / "out2")
    (model_dir / "weights.bin").write_bytes(b"456")
    assert hub.publish(model_dir, "user/model") != revision


def cli(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "vihermes", *map(str, args)],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


def test_cli_publish_and_fetch_with_a_local_hub(model_dir, tmp_path):
    env = {"VIHERMES_HUB_DIR": str(tmp_path / "hub")}
    published = cli("publish", "--path", model_dir, "--repo", "user/model", env=env)
    assert published.returncode == 0, published.stderr
    info = json.loads(published.stdout)
    fetched = cli(
        "fetch", "--repo", "user/model", "--revision", info["revision"],
        "--dest", tmp_path / "got", env=env,
    )  # fmt: skip
    assert fetched.returncode == 0, fetched.stderr
    assert (tmp_path / "got" / "weights.bin").exists()


def test_cli_rejects_an_unpinned_revision(tmp_path):
    result = cli("fetch", "--repo", "user/model", "--revision", "main", "--dest", tmp_path)
    assert result.returncode == 2 and "commit hash" in result.stderr
