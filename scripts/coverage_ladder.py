# This file was generated with the assistance of an AI coding tool.
"""Coverage ladder — per-module floors that ratchet with the OCC-exit milestones.

Run after a coverage-producing pytest session in the environment that matches
the run (the a320 conda env for oracle-inclusive numbers):

    pytest tests/ --cov -q
    python -m coverage json -o /tmp/geometrykit-cov.json
    python scripts/coverage_ladder.py /tmp/geometrykit-cov.json

Floors are set at today's measured value minus a small drift allowance; the
``planned`` column documents the milestone that is expected to raise each
floor next (PLAN-occ-exit.md §5). A red floor = someone deleted tests or a
feature regressed; below the planned jump is fine — regressions are not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOLERANCE = 0.5  # percentage points of allowed noise around a floor

#: module-relative path → (floor %, planned-milestone note)
FLOORS: dict[str, tuple[float, str]] = {
    "geometrykit/geometry/primitives.py": (75.0, "M2+ eval core"),
    "geometrykit/geometry/path.py": (59.0, "M2+ divide/offset paths"),
    "geometrykit/geometry/patterns.py": (95.0, "M2 gate removal"),
    "geometrykit/geometry/biarc.py": (84.0, "M3+ interpolation"),
    "geometrykit/geometry/frames.py": (52.0, "M6 sweep frames"),
    "geometrykit/geometry/subdivision.py": (92.0, "stable"),
    "geometrykit/geometry/curve.py": (39.0, "M3 interpolation + M4 projection"),
    "geometrykit/geometry/intersection.py": (36.0, "M5 section walk"),
    "geometrykit/geometry/surface.py": (26.0, "M5 tessellate + sections (tess/clip kernels)"),
    "geometrykit/geometry/spiral.py": (61.0, "M4+ projection"),
    "geometrykit/geometry/transform.py": (61.0, "M6+ loft"),
    "geometrykit/geometry/field.py": (94.0, "M2 landed — gate-free baseline"),
    "geometrykit/contract.py": (92.0, "complete"),
    "geometrykit/engine.py": (90.0, "complete"),
}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    data = json.loads(Path(sys.argv[1]).read_text())
    raw_files = data.get("files", {})

    def lookup(module: str):
        """Coverage json keys are repo-root-relative; match on the suffix."""
        for path, entry in raw_files.items():
            if path.endswith(module):
                return entry
        return None

    failures = 0
    total_of = data.get("totals", {})
    print(f"total coverage: {total_of.get('percent_covered', 0):.2f}%"
          f"  (ladder: rises per milestone)")
    for module, (floor, note) in sorted(FLOORS.items()):
        entry = lookup(module)
        if entry is None:
            print(f"MISSING {module}: not in coverage report (renamed? removed?)")
            failures += 1
            continue
        pct = entry["summary"]["percent_covered"]
        status = "ok" if pct >= floor - TOLERANCE else "REGRESSION"
        if pct < floor - TOLERANCE:
            failures += 1
        planned_jump = ""
        if pct < floor + 0.5:
            planned_jump = f"  [ladder: {note}]"
        print(
            f"{status:>9}  {module:<44} {pct:6.2f}%  (floor {floor:.1f}%){planned_jump}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
