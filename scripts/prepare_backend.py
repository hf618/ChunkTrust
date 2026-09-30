#!/usr/bin/env python3
"""Create a pinned upstream checkout and apply the released source overlay.

Existing directories are refused. Upstream experiment trees are never changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def prepare(name, destination, local_source=None, heldout=False):
    catalog = json.loads((ROOT / "third_party/upstreams.json").read_text())
    spec = catalog[name]
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}; choose a new directory")
    subprocess.run(["git", "clone", "--no-checkout", str(local_source or spec["url"]), str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", spec["commit"]], check=True)
    overlays = [name]
    if heldout:
        if name != "robotwin":
            raise ValueError("held-out QHA overlay requires robotwin")
        overlays.append("qha_heldout")
    installed = []
    for overlay in overlays:
        source = ROOT / "third_party" / "overlays" / overlay
        for p in sorted(source.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:
                continue
            target = destination / p.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            installed.append({"path": str(target.relative_to(destination)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    (destination / ".chunktrust-install.json").write_text(json.dumps({"upstream": spec, "overlays": overlays, "files": installed}, indent=2)+'\n')
    print(f"Prepared {name} at {destination}; install its environment following docs/backends.md")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=json.loads((ROOT/'third_party/upstreams.json').read_text()))
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--local-source', type=Path, help='optional existing clone; only its committed Git objects are used')
    parser.add_argument('--heldout', action='store_true')
    args = parser.parse_args()
    if args.heldout and args.backend != 'robotwin':parser.error('--heldout requires robotwin')
    prepare(args.backend, args.destination, args.local_source, args.heldout)
