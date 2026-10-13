# This file was generated with the assistance of an AI coding tool.
"""Capture the A320 demo corpus as geometrykit contract fixtures.

Run where ifckit and its dependencies are importable (the a320 conda env):

    /opt/miniconda3/envs/a320-ifc-configurator/bin/python scripts/capture_corpus.py

Executes selected demo projects through the same code path the configurator
uses (ast-parsed INPUTS + run(**inputs)), and snapshots every element dict
as a contract v0 record (``{id, kind, body, sig}``) into
``tests/fixtures/corpus/<project>.json``. The committed snapshots are the
M-phase fixture corpus: kernel tests, oracle goldens, and parity runs all
consume them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A320_ROOT = Path("/Users/Mauc/A320-ifc-configurator")
A320_BACKEND_SRC = A320_ROOT / "backend/src"

PROJECTS = ["flowing-roof", "bridge-example", "muur-met-raam"]


def load_inputs(project_id: str) -> dict:
    import ast

    model_path = Path("/Users/Mauc/A320-ifc-configurator/projecten") / project_id / "model.py"
    tree = ast.parse(model_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "INPUTS":
                    return {k: v.get("default") for k, v in ast.literal_eval(node.value).items()}
    return {}


def run_project(project_id: str):
    import contextlib
    import importlib.util
    import io

    if str(A320_BACKEND_SRC) not in sys.path:
        sys.path.insert(0, str(A320_BACKEND_SRC))
    if str(A320_ROOT) not in sys.path:  # models import the repo-top `helpers` package
        sys.path.insert(0, str(A320_ROOT))

    model_path = Path("/Users/Mauc/A320-ifc-configurator/projecten") / project_id / "model.py"
    spec = importlib.util.spec_from_file_location(f"projecten.{project_id}.model", str(model_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    inputs = load_inputs(project_id)
    kwargs = {k: inputs.get(k) for k in inputs}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = mod.run(**kwargs)
    if not isinstance(result, list):
        result = [result]
    return inputs, result


def main() -> None:
    import importlib.util

    if str(A320_BACKEND_SRC) not in sys.path:
        sys.path.insert(0, str(A320_BACKEND_SRC))

    out_dir = ROOT / "tests" / "fixtures" / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the A320 backend's Pending-to-dict serializer (a320 env has fastapi).
    ser_path = A320_BACKEND_SRC / "routes" / "projects.py"
    sspec = importlib.util.spec_from_file_location("a320_projects_serialize", str(ser_path))
    assert sspec is not None and sspec.loader is not None
    smod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(smod)
    serialize = smod._serialize_element

    for project_id in PROJECTS:
        inputs, result = run_project(project_id)
        elements, helpers = [], []
        for item in result:
            if isinstance(item, dict) and "__type__" in item:
                helpers.append(item)
            elif isinstance(item, dict) and "element" in item:
                rec = {"kind": "element", "body": serialize(item["element"]),
                       "id": item.get("id")}
                elements.append(rec)
            else:
                body = serialize(item)
                elements.append({"kind": "element", "body": body, "id": body.get("id")})
        snapshot = {
            "project": project_id,
            "inputs": inputs,
            "elements": elements,
            "helpers": helpers,
        }
        path = out_dir / f"{project_id}.json"
        path.write_text(json.dumps(snapshot, indent=1, default=str))
        print(f"{project_id}: {len(elements)} elements, {len(helpers)} helpers → {path.name}")


if __name__ == "__main__":
    main()
