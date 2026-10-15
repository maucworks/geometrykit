#!/usr/bin/env python3
# This file was generated with the assistance of an AI coding tool.
"""Build the ``geometrykit._rust`` accelerator extension into a venv.

Runs maturin against ``crates/geometrykit-py``, then copies the built
``_rust.abi3.so`` into the ``geometrykit`` package directory of the
target interpreter (default: this script's interpreter). Distribution
packaging of the extension is a later decision; until then this is the
supported dev route::

    python scripts/build_rust_extension.py           # this interpreter
    /path/to/venv/bin/python scripts/build_rust_extension.py

Requires: a Rust toolchain (cargo) and ``maturin`` importable from the
target interpreter (``pip install "maturin>=1,<2"``).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY_CRATE = REPO / "crates" / "geometrykit-py"


def _venv_bin(python: str) -> Path:
    # sys.prefix is the venv root for a venv interpreter; resolving
    # sys.executable would follow the python symlink to the base env.
    return Path(sys.prefix) / "bin"


def main() -> int:
    python = sys.executable
    maturin = _venv_bin(python) / "maturin"
    if not maturin.exists():
        maturin = shutil.which("maturin")  # type: ignore[assignment]
    if maturin is None:
        print("maturin not found: pip install 'maturin>=1,<2' first", file=sys.stderr)
        return 2

    out_dir = REPO / "wip" / "wheels"
    cmd = [
        str(maturin), "build", "--release",
        "-m", str(PY_CRATE / "Cargo.toml"),
        "-o", str(out_dir),
    ]
    env = dict(**__import__("os").environ)
    venv_prefix = _venv_bin(python)
    if (venv_prefix / "pyvenv.cfg").exists():
        env.setdefault("VIRTUAL_ENV", str(venv_prefix.parent))
        env["PATH"] = f"{venv_prefix}{__import__('os').pathsep}{env['PATH']}"
    print("building:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, env=env, check=True)

    wheels = sorted(out_dir.glob("geometrykit_rust-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        print("no wheel produced", file=sys.stderr)
        return 2

    import geometrykit  # the target interpreter's own install

    pkg_dir = Path(geometrykit.__file__).resolve().parent
    with zipfile.ZipFile(wheels[-1]) as zf:
        libs = [n for n in zf.namelist() if n.endswith((".so", ".pyd"))]
        if not libs:
            print(f"{wheels[-1]} contains no extension module", file=sys.stderr)
            return 2
        target = pkg_dir / libs[0].split("/")[-1]
        with zf.open(libs[0]) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    print(f"installed extension: {target}")
    probe = subprocess.run(
        [python, "-c",
         "import geometrykit._rust as r, geometrykit.engine as e;"
         " print(r.core_version(), '|', e.engine())"],
        capture_output=True, text=True,
    )
    print(probe.stdout.strip() or probe.stderr.strip())
    return probe.returncode


if __name__ == "__main__":
    raise SystemExit(main())
