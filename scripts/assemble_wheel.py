#!/usr/bin/env python3
# This file was generated with the assistance of an AI coding tool.
"""Assemble the binary accelerator wheel from the pure wheel + core crate.

Route (per platform):
  1. build the pure wheel via the ordinary setuptools route
     (``python -m build --wheel``), which is the universal fallback;
  2. compile ``crates/geometrykit-py`` with cargo (release);
  3. inject the compiled ``_rust.abi3.so`` into ``geometrykit/`` inside
     the wheel, retag it (``cp39``/``abi3``/platform), repack (the
     ``wheel`` tool rebuilds RECORD), leaving a wheel whose engine
     selection flips to ``rust`` on exactly matching platforms.

Used unchanged both locally and in CI; the pure fallback wheel ships
together with the binary matrix.

    python scripts/assemble_wheel.py \
        --wheel-wheels-dir wip/dist \
        --wheel-abi macosx_11_0_arm64 \
        --out wip/wheelhouse

Needs: ``pip install build wheel`` and the `wheel` CLI (bundled).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CRATE = REPO / "crates" / "geometrykit-py"


def _run(cmd: list, cwd: Path | None = None) -> None:
    print("run:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def build_extension() -> Path:
    """Release-build the core-in-python module: returns the .so copy."""
    _run(["cargo", "build", "--release", "--quiet"], cwd=CRATE)
    release = CRATE.parent / "target" / "release"
    if sys.platform == "win32":
        dylib, so_name = release / "_rust.dll", "_rust.pyd"
    elif sys.platform == "darwin":
        dylib, so_name = release / "lib_rust.dylib", "_rust.abi3.so"
    else:
        dylib, so_name = release / "lib_rust.so", "_rust.abi3.so"
    if not dylib.exists():
        raise SystemExit(f"cargo produced no dylib at {dylib}")
    out = Path(sys.argv[0]).resolve().parent.parent / "wip"
    out.mkdir(exist_ok=True)
    copy = out / so_name
    shutil.copy2(dylib, copy)
    return copy


def wheel_build(repo: Path, out_dir: Path) -> Path:
    """Build the pure fallback wheel via setuptools; return its path."""
    _run([sys.executable, "-m", "build", "--wheel", "-o", str(out_dir)], cwd=repo)
    wheels = sorted(out_dir.glob("geometrykit-*.whl"))
    if not wheels:
        raise SystemExit("no pure wheel produced")
    return wheels[-1]


def _records_root(entries: list) -> None:
    """Rewrite RECORD in-place: sha256 urlsafe digests, own line bare."""
    hashes = {}
    for file in entries:
        if file.name.endswith("RECORD"):
            continue
        digest = hashlib.sha256(file.read_bytes()).digest()
        hashes[file] = (
            base64.urlsafe_b64encode(digest).decode().rstrip("="),
            file.stat().st_size,
        )
    return hashes


def assemble(pure: Path, so: Path, tag: str, out_dir: Path) -> Path:
    """Inject the .so, retag, repack with a fresh RECORD (zip surgery)."""
    with zipfile.ZipFile(pure) as src:
        names = src.namelist()
        record_name = next(
            n for n in names if n.endswith(".dist-info/RECORD")
        )
        entries: list[tuple[str, bytes]] = []
        for info in src.infolist():
            if info.filename.endswith("/WHEEL"):
                body = src.read(info).decode()
                body = "\n".join(
                    ln for ln in body.splitlines() if not ln.startswith("Tag:")
                )
                body += f"\nTag: cp39-abi3-{tag}\n"
                entries.append((info.filename, body.encode()))
            elif info.filename != record_name:
                entries.append((info.filename, src.read(info)))
    entries.append(("geometrykit/" + so.name, so.read_bytes()))

    content = [
        (name, data, hashlib.sha256(data).digest(), len(data))
        for name, data in entries
    ]

    out_name = pure.name.replace("py3-none-any", f"cp39-abi3-{tag}")
    dest = out_dir / out_name
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as dst:
        for name, data, _digest, _size in content:
            dst.writestr(name, data)
        record = "\n".join(
            f"{name},sha256="
            f"{base64.urlsafe_b64encode(digest).decode().rstrip('=')},"
            f"{size}"
            for name, _data, digest, size in content
        )
        dst.writestr(record_name, record + "\n")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", default="wip/dist")
    parser.add_argument(
        "--platform-tag", required=True,
        help="the wheel platform tag, e.g. macosx_11_0_arm64",
    )
    parser.add_argument("--out-dir", default="wip/wheelhouse")
    args = parser.parse_args()
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pure = wheel_build(REPO, out_dir)
    so = build_extension()
    result = assemble(pure, so, args.platform_tag, out_dir)
    print(f"assembled: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
