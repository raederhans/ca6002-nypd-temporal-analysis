"""Build and execute the reader-facing Part 3 notebook.

The notebook re-runs only fast, deterministic pieces (class balance, a small
leakage demonstration) and displays the training outputs produced by
train_predictive_model.py; full training is intentionally not repeated here.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_ai_algorithm.ipynb"


def build_notebook(output_path: Path = NOTEBOOK_PATH) -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Part 3 — AI Algorithm: predicting recorded arrest severity from context

This notebook reproduces the Person 3 analysis. A Random Forest predicts whether a recorded arrest is a Felony, Misdemeanor or Violation from contextual characteristics only — borough, precinct, age group, month, day of week, jurisdiction and coordinates. Fields that directly encode the charge (LAW_CODE, PD_DESC, OFNS_DESC) are excluded so the answer cannot leak into the inputs. Arrest records are recorded enforcement activity, not crime incidence."""),
        nbf.v4.new_code_cell("""from pathlib import Path
import sys
import json
import pandas as pd
from IPython.display import display, Image

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from train_predictive_model import load_data, check_no_leakage, make_pipeline, SEVERITY_ORDER, SEVERITY_LABELS, CONTEXT_FEATURES, RANDOM_STATE

DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'nypd_arrests_clean.csv'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'part3'
FIGURE_DIR = PROJECT_ROOT / 'figures' / 'part3'
frame = load_data(DATA_PATH)
len(frame)"""),
        nbf.v4.new_markdown_cell("## 1. Target and class balance"),
        nbf.v4.new_code_cell("""severity = frame['LAW_CAT_CD'].astype('string').str.strip().str.upper()
core = frame[severity.isin(SEVERITY_ORDER)].copy()
balance = severity[severity.isin(SEVERITY_ORDER)].value_counts().to_frame('records')
balance['share_pct'] = (100 * balance['records'] / balance['records'].sum()).round(2)
balance.index = [SEVERITY_LABELS[c] for c in balance.index]
print(f"Core labelled records: {len(core):,} of {len(frame):,}")
print(f"Excluded (missing or non-core codes): {len(frame) - len(core):,}")
display(balance)"""),
        nbf.v4.new_markdown_cell("""Misdemeanor is the majority class at 59.0%, Violation is rare at 1.35%. A model that always predicts Misdemeanor scores 59.0% accuracy without learning anything, so the pipeline uses `class_weight=\"balanced\"` and macro-F1 as the tuning metric, and the majority-class guess is the reference baseline."""),
        nbf.v4.new_markdown_cell("## 2. Leakage-aware feature design"),
        nbf.v4.new_code_cell("""# Quick demonstration on a 4,000-record sample: adding charge-encoding fields inflates accuracy
from sklearn.model_selection import train_test_split
from train_predictive_model import build_feature_frame, CATEGORICAL_FEATURES, LEAKAGE_COLUMNS

sample = core.sample(4000, random_state=RANDOM_STATE)
y = sample['LAW_CAT_CD'].astype('string').str.strip().str.upper()
s_train, s_test, ys_train, ys_test = train_test_split(sample, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

variants = [
    ('Contextual', build_feature_frame(s_train), build_feature_frame(s_test), list(CATEGORICAL_FEATURES)),
    ('With charge fields', build_feature_frame(s_train, include_leakage=True), build_feature_frame(s_test, include_leakage=True),
     list(CATEGORICAL_FEATURES) + list(LEAKAGE_COLUMNS[:3])),
]
for name, features, test_features, cats in variants:
    pipe = make_pipeline(categorical=cats, model_params={'n_estimators': 50, 'max_depth': 15})
    pipe.fit(features, ys_train)
    acc = (pipe.predict(test_features) == ys_test.to_numpy()).mean()
    print(f"{name:18s} sample holdout accuracy: {acc:.3f}")

check_no_leakage(list(CONTEXT_FEATURES))
print('CONTEXT_FEATURES contain no leakage columns - verified.')"""),
        nbf.v4.new_markdown_cell("""The two models differ only in the input fields. Adding LAW_CODE, PD_CD and OFNS_DESC — fields that already encode the charge — pushes accuracy toward 1.0 without learning anything about context. The reported model deliberately excludes those fields; this exclusion is itself a design finding."""),
        nbf.v4.new_markdown_cell("## 3. Holdout performance"),
        nbf.v4.new_code_cell("""metrics = json.loads((OUTPUT_DIR / 'model_metrics.json').read_text())
print(json.dumps({k: v for k, v in metrics.items() if k != 'confusion_matrix'}, indent=2))
display(pd.read_csv(OUTPUT_DIR / 'classification_report.csv'))
display(pd.read_csv(OUTPUT_DIR / 'confusion_matrix.csv'))"""),
        nbf.v4.new_code_cell("display(Image(filename=str(FIGURE_DIR / '01_confusion_matrix.png'), width=1000))"),
        nbf.v4.new_markdown_cell("""Felony and Misdemeanor records are separated reliably. Violation predictions are imprecise: with 1.35% prevalence and balanced class weights the model flags too many records as Violation, so precision is only 0.181 while recall is 0.765. That honest gap is reported rather than hidden by quoting overall accuracy alone."""),
        nbf.v4.new_markdown_cell("## 4. Parameter fine tuning"),
        nbf.v4.new_code_cell("""grid = pd.read_csv(OUTPUT_DIR / 'grid_search_results.csv')
display(grid)
print((OUTPUT_DIR / 'best_parameters.json').read_text())"""),
        nbf.v4.new_markdown_cell("""Twelve combinations of `n_estimators`, `max_depth` and `min_samples_leaf` were evaluated with stratified 5-fold cross-validation on a 30,000-row subsample of the training split, scored by macro-F1. The holdout set was never touched during tuning; the chosen parameters were then refitted on all 112,380 training records."""),
        nbf.v4.new_markdown_cell("## 5. Explainability — feature importance"),
        nbf.v4.new_code_cell("""importance = pd.read_csv(OUTPUT_DIR / 'feature_importance.csv')
display(importance)
display(Image(filename=str(FIGURE_DIR / '02_feature_importance.png'), width=1000))"""),
        nbf.v4.new_markdown_cell("""Mean decrease in impurity, aggregated from one-hot categories back to the eight model inputs. Importance ranks describe the fitted model's decisions; they do not rank societal mechanisms."""),
        nbf.v4.new_markdown_cell("## 6. Explainability — partial dependence"),
        nbf.v4.new_code_cell("""pdp = pd.read_csv(OUTPUT_DIR / 'pdp_values.csv')
display(pdp)
display(Image(filename=str(FIGURE_DIR / '03_partial_dependence.png'), width=1000))"""),
        nbf.v4.new_markdown_cell("""The predicted Felony probability falls steadily from the under-18 group to older groups. Borough and month shift the probability only modestly. Partial dependence summarises the fitted model's association and does not establish a causal relationship."""),
        nbf.v4.new_markdown_cell("## 7. Learning curve"),
        nbf.v4.new_code_cell("""curve = pd.read_csv(OUTPUT_DIR / 'learning_curve.csv')
display(curve)
display(Image(filename=str(FIGURE_DIR / '04_learning_curve.png'), width=1000))"""),
        nbf.v4.new_markdown_cell("""Validation macro-F1 keeps improving as training records are added, with no sign of a plateau at the largest training size of 74,920 records; the gap to the training score shows the model is not overfitting hard at these parameters."""),
        nbf.v4.new_markdown_cell("## 8. The leakage warning"),
        nbf.v4.new_code_cell("""leakage = pd.read_csv(OUTPUT_DIR / 'leakage_comparison.csv')
display(leakage)
display(Image(filename=str(FIGURE_DIR / '05_leakage_comparison.png'), width=1000))"""),
        nbf.v4.new_markdown_cell("""The leakage-augmented model exists only as a comparison: it shows how easily an artificially inflated accuracy could be presented as progress. Its score must not be cited as model performance."""),
        nbf.v4.new_markdown_cell("""## 9. Interpretation boundary

- Arrest records reflect policing, reporting and enforcement processes; the model describes recorded arrest activity, not crime incidence, and does not predict future events.
- The six-month snapshot (Jan–Jun 2026) is not evidence of longer-term stability.
- Feature importance and partial dependence describe the fitted model; they do not establish causal relationships.
- The Violation class is too rare for the model to separate; this limitation is stated alongside every score.""") ,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT)}})
    executed = client.execute()
    nbf.write(executed, output_path)
    return output_path


if __name__ == "__main__":
    print(build_notebook())
