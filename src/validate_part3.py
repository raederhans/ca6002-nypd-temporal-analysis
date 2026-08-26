"""Independent validation of Part 3 model outputs, figures, notebook, and language."""
from __future__ import annotations

import json
from pathlib import Path
import re

import nbformat
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURE_STEMS = ("01_confusion_matrix", "02_feature_importance", "03_partial_dependence", "04_learning_curve", "05_leakage_comparison")
DATA_ARTIFACTS = (
    "model_metrics.json", "classification_report.csv", "confusion_matrix.csv", "feature_importance.csv",
    "grid_search_results.csv", "best_parameters.json", "learning_curve.csv", "leakage_comparison.csv",
    "pdp_values.csv", "analysis_metadata.json",
)
DOC_ARTIFACTS = ("key_findings.md", "methodology.md", "limitations.md", "slide_plan.md", "team_handoff.md")


def validate(project_root: Path = PROJECT_ROOT) -> list[str]:
    out = project_root / "outputs" / "part3"
    figs = project_root / "figures" / "part3"
    data = pd.read_csv(project_root / "data" / "processed" / "nypd_arrests_clean.csv", low_memory=False)
    checks = []
    assert len(data) == 141870; checks.append("PASS — processed row count is 141,870")
    severity = data["LAW_CAT_CD"].astype("string").str.strip().str.upper()
    core = int(severity.isin(["F", "M", "V"]).sum())
    assert core == 140476 and len(data) - core == 1394; checks.append("PASS — core F/M/V denominator is 140,476 and 1,394 rows are excluded")

    meta = json.loads((out / "analysis_metadata.json").read_text())
    assert meta["train_size"] == 112380 and meta["test_size"] == 28096; checks.append("PASS — stratified 80/20 split is 112,380 / 28,096")
    assert meta["train_size"] + meta["test_size"] == meta["core_fmv_records"] == 140476; checks.append("PASS — split reconciles to the core denominator")
    assert meta["random_state"] == 42; checks.append("PASS — random state 42 is fixed")

    for name in DATA_ARTIFACTS + DOC_ARTIFACTS:
        assert (out / name).is_file(), name
    checks.append("PASS — all ten data artifacts and five handoff documents exist")

    metrics = json.loads((out / "model_metrics.json").read_text())
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "per_class", "confusion_matrix"):
        assert key in metrics
    assert len(metrics["per_class"]) == 3 and len(metrics["confusion_matrix"]) == 3 and all(len(r) == 3 for r in metrics["confusion_matrix"])
    assert 0.25 < metrics["macro_f1"] < 0.75 and metrics["balanced_accuracy"] > 0.30
    assert abs(metrics["majority_baseline_accuracy"] - 0.5903) < 0.002
    assert metrics["leakage_accuracy"] > metrics["accuracy"] + 0.05
    assert metrics["accuracy"] >= metrics["majority_baseline_accuracy"] - 0.05
    checks.append("PASS — holdout metrics are in range and the leakage model inflates accuracy by more than 0.05")

    cm = pd.read_csv(out / "confusion_matrix.csv", index_col=0)
    assert int(cm.to_numpy().sum()) == 28096 and (cm.sum(axis=1) > 0).all(); checks.append("PASS — confusion matrix row sums reconcile to 28,096")
    report = pd.read_csv(out / "classification_report.csv", index_col=0)
    for label in ("Felony", "Misdemeanor", "Violation"):
        assert abs(report.loc[label, "precision"] - metrics["per_class"][label]["precision"]) < 0.0005
        assert abs(report.loc[label, "recall"] - metrics["per_class"][label]["recall"]) < 0.0005
    checks.append("PASS — classification report matches the per-class metrics JSON")

    grid = pd.read_csv(out / "grid_search_results.csv")
    assert len(grid) == 12; checks.append("PASS — grid search contains exactly 12 parameter combinations")
    best = json.loads((out / "best_parameters.json").read_text())
    top = grid.iloc[0]
    assert int(top["n_estimators"]) == best["n_estimators"] and int(top["min_samples_leaf"]) == best["min_samples_leaf"]
    assert (pd.isna(top["max_depth"]) and best["max_depth"] is None) or (int(top["max_depth"]) == best["max_depth"])
    assert abs(float(top["mean_test_f1_macro"]) - best["best_cv_f1_macro"]) < 1e-9
    checks.append("PASS — rank-1 grid parameters equal the stored best parameters")

    importance = pd.read_csv(out / "feature_importance.csv")
    assert len(importance) == 8 and abs(importance["share_pct"].sum() - 100) < 0.01
    checks.append("PASS — feature importance aggregates back to eight inputs summing to 100%")

    curve = pd.read_csv(out / "learning_curve.csv")
    assert len(curve) == 5 and curve["val_mean"].between(0, 1).all() and curve["train_mean"].iloc[-1] > curve["val_mean"].iloc[-1]
    checks.append("PASS — learning curve has five sizes and a training/validation gap at full size")

    for stem in FIGURE_STEMS:
        png, svg = figs / f"{stem}.png", figs / f"{stem}.svg"
        assert png.is_file() and svg.is_file()
        assert Image.open(png).size == (2560, 1440)
        assert "<svg" in svg.read_text(encoding="utf-8")[:1000]
    checks.append("PASS — five figure families exist as 2560×1440 PNG and valid SVG")

    notebook = nbformat.read(project_root / "notebooks" / "03_ai_algorithm.ipynb", as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert code_cells and all(c.get("execution_count") is not None for c in code_cells)
    assert not any(o.get("output_type") == "error" for c in code_cells for o in c.get("outputs", []))
    checks.append("PASS — Part 3 notebook executed top-to-bottom without errors")

    docs = "\n".join(p.read_text(encoding="utf-8") for p in out.glob("*.md"))
    banned = [r"most crime", r"crime rate is", r"\bcauses?\b", r"\bproves?\b", r"predicts future", r"will offend", r"identif(?:y|ies) criminals"]
    assert not any(re.search(pattern, docs, re.I) for pattern in banned); checks.append("PASS — handoff language avoids unsupported crime/causal claims")
    key_findings = (out / "key_findings.md").read_text(encoding="utf-8")
    slide_plan = (out / "slide_plan.md").read_text(encoding="utf-8")
    assert "recorded arrests" in key_findings and "recorded arrests" in slide_plan; checks.append("PASS — key documents use the recorded-arrests framing")
    assert "artificially inflated" in slide_plan; checks.append("PASS — slide plan names the artificial leakage inflation")
    assert slide_plan.count("## Slide C") == 4; checks.append("PASS — slide plan contains four Part 3 slides")

    report = "# Part 3 Validation Report\n\nOverall assessment: **Ready for review**\n\n" + "\n".join(f"- {c}" for c in checks) + "\n"
    (out / "validation_report.md").write_text(report, encoding="utf-8")
    return checks


if __name__ == "__main__":
    for item in validate():
        print(item)
