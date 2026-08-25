"""Run the complete Part 2 pipeline without modifying Part 1 artifacts."""
from __future__ import annotations

from analyze_spatial_demographic import run_analysis
from build_part2_deliverables import build_deliverables
from build_part2_notebook import build_notebook
from validate_part2 import validate


def main() -> None:
    run_analysis()
    build_deliverables()
    build_notebook()
    checks = validate()
    print(f"Part 2 pipeline complete with {len(checks)} validation checks passed.")


if __name__ == "__main__":
    main()
