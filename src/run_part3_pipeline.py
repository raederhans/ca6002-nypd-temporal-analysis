"""Run the complete Part 3 pipeline without modifying Part 1 or Part 2 artifacts."""
from __future__ import annotations

from train_predictive_model import run_analysis
from build_part3_deliverables import build_deliverables
from build_part3_notebook import build_notebook
from validate_part3 import validate


def main() -> None:
    run_analysis()
    build_deliverables()
    build_notebook()
    checks = validate()
    print(f"Part 3 pipeline complete with {len(checks)} validation checks passed.")


if __name__ == "__main__":
    main()
