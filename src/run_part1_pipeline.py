"""Run the frozen-snapshot CA6002 Part 1 pipeline from cleaning through QA."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

import analyze_temporal
import build_notebook
import clean_nypd_data
import validate_part1
from build_deliverables import build_deliverables
from download_nypd_data import download_snapshot


def execute_notebook(path: Path, project_root: Path) -> None:
    """Execute a notebook from the project root and replace it atomically."""

    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(project_root)}},
    )
    client.execute(cwd=str(project_root))
    temporary = path.with_suffix(path.suffix + ".tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(path)
    build_notebook.validate_notebook(path, require_executed=True)


def run(
    project_root: Path,
    *,
    download: bool = False,
    execute: bool = True,
) -> bool:
    root = project_root.resolve()
    if download:
        download_snapshot(root)

    clean_nypd_data.run_pipeline(project_root=root)
    analyze_temporal.analyze_temporal(
        processed_csv=root / "data" / "processed" / "nypd_arrests_clean.csv",
        audit_csv=root / "outputs" / "part1" / "missingness_summary.csv",
        metadata_json=root / "outputs" / "part1" / "dataset_snapshot_metadata.json",
        outputs_dir=root / "outputs" / "part1",
        figures_dir=root / "figures" / "part1",
    )
    build_deliverables(project_root=root)
    notebook_path = build_notebook.build_notebook(project_root=root)
    if execute:
        execute_notebook(notebook_path, root)

    validator, summary = validate_part1.validate(root)
    validate_part1._write_report(root, validator, summary)
    return validator.passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch a new official snapshot before rebuilding downstream outputs.",
    )
    parser.add_argument(
        "--skip-notebook-execution",
        action="store_true",
        help="Generate but do not execute the notebook; final validation will remain incomplete.",
    )
    args = parser.parse_args()
    passed = run(
        args.project_root,
        download=args.download,
        execute=not args.skip_notebook_execution,
    )
    print("Part 1 pipeline: READY TO SHARE" if passed else "Part 1 pipeline: NEEDS REVISION")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
