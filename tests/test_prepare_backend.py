"""Exercise source preparation with real Git objects and no network or devices."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_backend.py"
spec = importlib.util.spec_from_file_location("prepare_backend", SCRIPT)
preparer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparer)


def test_layered_overlay_uses_committed_source_and_records_files(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    (source / "base.txt").write_text("committed\n")
    subprocess.run(["git", "-C", str(source), "add", "base.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "-c", "user.name=Test", "-c",
         "user.email=test@example.com", "commit", "-m", "fixture"],
        check=True, capture_output=True,
    )
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    (source / "base.txt").write_text("uncommitted local changes\n")
    (source / "private.txt").write_text("not exported\n")
    release = tmp_path / "release"
    overlays = release / "third_party/overlays"
    for name, content in [("shared", "base overlay\n"), ("device", "device overlay\n")]:
        folder = overlays / name
        folder.mkdir(parents=True)
        (folder / "settings.txt").write_text(content)
    catalog = {"fixture": {"url": "unused", "commit": revision, "overlays": ["shared", "device"]}}
    (release / "third_party/upstreams.json").write_text(json.dumps(catalog))
    monkeypatch.setattr(preparer, "ROOT", release)
    destination = tmp_path / "prepared"
    preparer.prepare("fixture", destination, local_source=source)
    assert (destination / "base.txt").read_text() == "committed\n"
    assert not (destination / "private.txt").exists()
    assert (destination / "settings.txt").read_text() == "device overlay\n"
    receipt = json.loads((destination / ".chunktrust-install.json").read_text())
    assert receipt["overlays"] == ["shared", "device"]
    assert receipt["upstream"]["commit"] == revision
    assert len(receipt["files"]) == 1  # The last layer replaces the earlier file and hash.
    assert receipt["files"][-1]["sha256"] == hashlib.sha256(b"device overlay\n").hexdigest()
    for record in receipt["files"]:
        assert hashlib.sha256((destination / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    with pytest.raises(ValueError, match="already exists"):
        preparer.prepare("fixture", destination, local_source=source)


def test_incompatible_heldout_request_does_not_create_destination(tmp_path):
    destination = tmp_path / "robodriver"
    with pytest.raises(ValueError, match="requires robotwin"):
        preparer.prepare("robodriver", destination, heldout=True)
    assert not destination.exists()
