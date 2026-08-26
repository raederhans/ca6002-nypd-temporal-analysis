"""Part 3 predictive modelling of the shared cleaned NYPD data.

Run from any directory with:
    python src/train_predictive_model.py

The module reads the Part 1 processed baseline, trains a Random Forest that
predicts recorded arrest severity LAW_CAT_CD (Felony / Misdemeanor /
Violation) from contextual features, tunes its parameters by grid search,
exports evaluation, learning-curve, feature-importance and partial-dependence
evidence, and trains one leakage-augmented comparison model to demonstrate
that charge-encoding fields artificially inflate accuracy.

Arrest records are treated as recorded enforcement activity, not crime rates,
and the model describes the recording process rather than future events.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# joblib/loky reads this before spawning workers; the default Windows temp
# path contains a non-ASCII user name and crashes parallel model selection.
os.environ.setdefault("JOBLIB_TEMP_FOLDER", "C:/Users/Public")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "nypd_arrests_clean.csv"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figures" / "part3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part3"

RANDOM_STATE = 42
TUNING_SUBSAMPLE_SIZE = 30000
TEST_SIZE = 0.2

SEVERITY_ORDER = ("F", "M", "V")
SEVERITY_LABELS = {"F": "Felony", "M": "Misdemeanor", "V": "Violation"}
BOROUGH_LABELS = {"B": "Bronx", "K": "Brooklyn", "M": "Manhattan", "Q": "Queens", "S": "Staten Island"}
BOROUGH_ORDER = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
AGE_ORDER = ["<18", "18-24", "25-44", "45-64", "65+"]
MONTH_ORDER = ["1", "2", "3", "4", "5", "6"]
MONTH_SHORT = {"1": "Jan", "2": "Feb", "3": "Mar", "4": "Apr", "5": "May", "6": "Jun"}

CONTEXT_FEATURES = (
    "ARREST_BORO", "ARREST_PRECINCT", "AGE_GROUP", "MONTH",
    "DAY_OF_WEEK_NUM", "JURISDICTION_CODE", "LATITUDE", "LONGITUDE",
)
CATEGORICAL_FEATURES = CONTEXT_FEATURES[:6]
NUMERIC_FEATURES = ("LATITUDE", "LONGITUDE")
LEAKAGE_COLUMNS = ("LAW_CODE", "PD_CD", "OFNS_DESC", "PD_DESC")
FEATURE_LABELS = {
    "ARREST_BORO": "Borough", "ARREST_PRECINCT": "Precinct", "AGE_GROUP": "Age group",
    "MONTH": "Month", "DAY_OF_WEEK_NUM": "Day of week", "JURISDICTION_CODE": "Jurisdiction",
    "LATITUDE": "Latitude", "LONGITUDE": "Longitude",
}

SEVERITY_COLORS = {"Felony": "#0072B2", "Misdemeanor": "#E69F00", "Violation": "#CC79A7"}
TEXT = "#1F2937"
MUTED = "#4B5563"
GRID = "#D9DEE3"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREY = "#9AA3AA"

F1_MACRO = make_scorer(lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro", zero_division=0))


def load_data(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    required = CONTEXT_FEATURES + LEAKAGE_COLUMNS + ("LAW_CAT_CD",)
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    return frame


def check_no_leakage(columns: object) -> None:
    overlap = sorted(set(columns) & set(LEAKAGE_COLUMNS))
    if overlap:
        raise ValueError("Leakage columns rejected: " + ", ".join(overlap))


def build_feature_frame(frame: pd.DataFrame, include_leakage: bool = False) -> pd.DataFrame:
    """Return the raw (unencoded) feature matrix.

    The contextual frame contains only the eight agreed contextual features.
    With include_leakage=True the three charge-encoding categorical fields are
    added for the comparison experiment only.
    """
    categorical = list(CATEGORICAL_FEATURES)
    if include_leakage:
        categorical = categorical + ["LAW_CODE", "PD_CD", "OFNS_DESC"]
    else:
        check_no_leakage(categorical)
    out = frame[list(categorical) + list(NUMERIC_FEATURES)].copy()
    for column in categorical:
        out[column] = out[column].astype("string")
    for column in NUMERIC_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def make_pipeline(categorical: object = None, model_params: dict | None = None, rf_n_jobs: int = 1) -> Pipeline:
    """One-hot encode categoricals, scale coordinates, classify with a balanced Random Forest."""
    cats = list(categorical) if categorical is not None else list(CATEGORICAL_FEATURES)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ("num", StandardScaler(), list(NUMERIC_FEATURES)),
    ])
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=rf_n_jobs, **(model_params or {}))
    return Pipeline([("pre", pre), ("rf", rf)])


def split_core(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the core F/M/V records 80/20 with stratification on severity."""
    core = frame[frame["LAW_CAT_CD"].astype("string").str.strip().str.upper().isin(SEVERITY_ORDER)].copy()
    y = core["LAW_CAT_CD"].astype("string").str.strip().str.upper()
    train_idx, test_idx = train_test_split(
        np.arange(len(core)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    return {
        "train": core.iloc[train_idx],
        "test": core.iloc[test_idx],
        "y_train": y.iloc[train_idx].reset_index(drop=True),
        "y_test": y.iloc[test_idx].reset_index(drop=True),
    }


def majority_baseline(y: pd.Series) -> float:
    return float(y.value_counts(normalize=True).max())


def tune_on_subsample(X_train: pd.DataFrame, y_train: pd.Series, subsample_size: int = TUNING_SUBSAMPLE_SIZE) -> tuple[GridSearchCV, pd.DataFrame]:
    """Grid-search the Random Forest on a stratified subsample of the training set.

    The test split is untouched. scoring is macro-F1 so that the rare Violation
    class counts equally rather than being swamped by the 59% majority class.
    """
    check_no_leakage(X_train.columns)
    idx, _ = train_test_split(
        np.arange(len(X_train)), train_size=subsample_size, random_state=RANDOM_STATE, stratify=y_train,
    )
    X_tune, y_tune = X_train.iloc[idx], y_train.iloc[idx]
    pipe = make_pipeline(rf_n_jobs=1)
    grid = GridSearchCV(
        pipe,
        param_grid={
            "rf__n_estimators": [200, 400],
            "rf__max_depth": [None, 15, 30],
            "rf__min_samples_leaf": [1, 5],
        },
        scoring=F1_MACRO,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    grid.fit(X_tune, y_tune)
    results = pd.DataFrame({
        "n_estimators": grid.cv_results_["param_rf__n_estimators"],
        "max_depth": grid.cv_results_["param_rf__max_depth"],
        "min_samples_leaf": grid.cv_results_["param_rf__min_samples_leaf"],
        "mean_test_f1_macro": grid.cv_results_["mean_test_score"],
        "std_test_f1_macro": grid.cv_results_["std_test_score"],
        "rank": grid.cv_results_["rank_test_score"],
        "mean_fit_seconds": grid.cv_results_["mean_fit_time"],
        "mean_score_seconds": grid.cv_results_["mean_score_time"],
    }).sort_values("rank").reset_index(drop=True)
    return grid, results


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=list(SEVERITY_ORDER))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, labels=list(SEVERITY_ORDER), zero_division=0)
    per_class = {
        SEVERITY_LABELS[code]: {
            "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(cm[i].sum()),
        }
        for i, code in enumerate(SEVERITY_ORDER)
    }
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
        "predictions": pred,
    }


def compute_learning_curve(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 5), cv=3,
        scoring=F1_MACRO, n_jobs=-1, shuffle=True, random_state=RANDOM_STATE,
    )
    return pd.DataFrame({
        "train_size": sizes,
        "train_mean": train_scores.mean(axis=1), "train_std": train_scores.std(axis=1),
        "val_mean": val_scores.mean(axis=1), "val_std": val_scores.std(axis=1),
    })


def compute_pdp(model: Pipeline, X_test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One-dimensional partial dependence of the Felony probability for three categorical features.

    The estimator is a Pipeline fitted on the raw DataFrame, which is the
    sklearn 1.4 pattern for categorical partial dependence. Violation (1.35%)
    is omitted because its curves are noisy; see the Part 3 limitations.
    """
    felon_index = list(model.classes_).index("F")
    frames = {}
    for feature in ("AGE_GROUP", "ARREST_BORO", "MONTH"):
        result = partial_dependence(
            model, X_test, features=[feature], categorical_features=[feature],
            response_method="predict_proba",
        )
        values = [str(v) for v in result["grid_values"][0]]
        curve = result["average"][felon_index]
        data = pd.DataFrame({"category": values, "p_felony": curve})
        if feature == "ARREST_BORO":
            data["category"] = data["category"].map(BOROUGH_LABELS)
            data = data.set_index("category").reindex(BOROUGH_ORDER).reset_index()
        elif feature == "AGE_GROUP":
            data = data.set_index("category").reindex(AGE_ORDER).reset_index()
        else:
            data = data.set_index("category").reindex(MONTH_ORDER).reset_index()
        data["display"] = data["category"].map(MONTH_SHORT) if feature == "MONTH" else data["category"]
        frames[feature] = data
    return frames


def compute_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Mean-decrease-in-impurity importance, summed from one-hot dummies back to the eight raw features."""
    cat = model.named_steps["pre"].named_transformers_["cat"]
    num = model.named_steps["pre"].named_transformers_["num"]
    names = list(cat.get_feature_names_out()) + list(num.get_feature_names_out())
    importance = dict(zip(names, model.named_steps["rf"].feature_importances_))
    rows = []
    for feature in CONTEXT_FEATURES:
        total = sum(value for name, value in importance.items() if name.startswith(feature + "_") or name == feature)
        rows.append({"feature": FEATURE_LABELS[feature], "raw_feature": feature, "importance": float(total)})
    frame = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    frame["share_pct"] = 100 * frame["importance"] / frame["importance"].sum()
    return frame


def train_leakage_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, best_params: dict) -> dict:
    """Comparison model that additionally sees charge-encoding fields, for the leakage demonstration only."""
    Xl_train = build_feature_frame(X_train, include_leakage=True)
    Xl_test = build_feature_frame(X_test, include_leakage=True)
    pipe = make_pipeline(
        categorical=list(CATEGORICAL_FEATURES) + ["LAW_CODE", "PD_CD", "OFNS_DESC"],
        model_params=best_params, rf_n_jobs=-1,
    )
    pipe.fit(Xl_train, y_train)
    return evaluate_model(pipe, Xl_test, y_test)


# ---------------------------------------------------------------- figures


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.tick_params(colors=MUTED, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, figure_dir: Path) -> None:
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = [SEVERITY_LABELS[c] for c in SEVERITY_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    fig.subplots_adjust(left=.07, right=.89, top=.80, bottom=.14, wspace=.22)
    panels = [(cm, "Counts", "{:,}"), (cm_pct, "Row share (%)", "{:.1f}")]
    for ax, (matrix, panel, fmt) in zip(axes, panels):
        image = ax.imshow(matrix, cmap="Blues", aspect="auto")
        threshold = matrix.max() * 0.6
        for i in range(3):
            for j in range(3):
                ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center", fontsize=13, weight="bold",
                        color="white" if matrix[i, j] > threshold else TEXT)
        ax.set_xticks(range(3), [f"Predicted\n{l}" for l in labels], fontsize=11)
        ax.set_yticks(range(3), [f"Recorded\n{l}" for l in labels], fontsize=11)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(panel, loc="left", fontsize=13, color=MUTED, weight="bold", pad=8)
        fig.colorbar(image, ax=ax, fraction=.045, pad=.02, shrink=.85).ax.tick_params(colors=MUTED, labelsize=9)
    fig.text(.07, .955, "Felony and Misdemeanor separate; Violation predictions rarely are correct",
             fontsize=20, color=TEXT, weight="bold", ha="left")
    fig.text(.07, .91, "28,096-record holdout set; row-normalised shares on the right", color=MUTED, fontsize=11, ha="left")
    fig.text(.07, .05, "Predicted severity of recorded arrests — a descriptive model of enforcement activity, not a prediction of future events.", fontsize=9, color=MUTED, ha="left")
    _save(fig, figure_dir / "01_confusion_matrix")


def plot_feature_importance(importance: pd.DataFrame, figure_dir: Path) -> None:
    data = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.subplots_adjust(left=.14, right=.97, top=.76, bottom=.17)
    y = np.arange(len(data))
    ax.barh(y, data["share_pct"], color=BLUE, height=.62)
    for yi, value in zip(y, data["share_pct"]):
        ax.text(value + .8, yi, f"{value:.1f}%", va="center", fontsize=11, color=TEXT)
    ax.set_yticks(y, data["feature"]); ax.set_xlabel("Mean decrease in impurity (share of total)", color=MUTED)
    ax.set_xlim(0, data["share_pct"].max() * 1.15); ax.grid(axis="x", color=GRID, linewidth=.7)
    ax.spines[["top", "right", "left"]].set_visible(False); _style_axis(ax)
    fig.text(.14, .955, "Location and jurisdiction context drive the model's decisions", fontsize=20, color=TEXT, weight="bold", ha="left")
    fig.text(.14, .91, f"Feature importances of the tuned Random Forest, aggregated from one-hot categories to the {len(data)} model inputs", color=MUTED, fontsize=11, ha="left")
    fig.text(.14, .05, "Importance measures contribution to the fitted model only; it does not rank societal causes.", fontsize=9, color=MUTED, ha="left")
    _save(fig, figure_dir / "02_feature_importance")


def plot_partial_dependence(pdp: dict[str, pd.DataFrame], felony_share: float, figure_dir: Path) -> None:
    panels = [("AGE_GROUP", "Age group", pdp["AGE_GROUP"], pdp["AGE_GROUP"]["display"]),
              ("ARREST_BORO", "Borough", pdp["ARREST_BORO"], pdp["ARREST_BORO"]["display"]),
              ("MONTH", "Month", pdp["MONTH"], pdp["MONTH"]["display"])]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 7.2))
    fig.subplots_adjust(left=.08, right=.98, top=.78, bottom=.18, wspace=.40)
    for ax, (_, xlabel, data, ticks) in zip(axes, panels):
        x = np.arange(len(data))
        ax.plot(x, data["p_felony"], color=BLUE, marker="o", markersize=8, linewidth=2.5)
        for xi, value in zip(x, data["p_felony"]):
            ax.text(xi, value + .016, f"{value:.3f}", ha="center", fontsize=9, color=BLUE)
        ax.axhline(felony_share, color=GRID, linestyle="--", linewidth=1.4)
        ax.set_xticks(x, ticks, fontsize=10); ax.set_xlabel(xlabel, color=MUTED)
        ax.set_xlim(-.55, len(data) - .45); ax.set_ylim(felony_share - .02, data["p_felony"].max() + .05)
        ax.grid(axis="y", color=GRID, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False); _style_axis(ax)
    axes[0].set_ylabel("Predicted probability of Felony", color=MUTED)
    fig.text(.08, .955, "The under-18 group carries the highest predicted Felony probability", fontsize=20, color=TEXT, weight="bold", ha="left")
    fig.text(.08, .908, "Partial dependence on age group, borough and month; the dashed line marks the overall Felony share in the holdout set", color=MUTED, fontsize=11, ha="left")
    fig.text(.08, .055, "Partial dependence summarises the fitted model's association; it does not establish a causal effect.", fontsize=9, color=MUTED, ha="left")
    _save(fig, figure_dir / "03_partial_dependence")


def plot_learning_curve(curve: pd.DataFrame, figure_dir: Path) -> None:
    """Learning curve with a broken y-axis: the flat training curve (≈0.62) and
    the slowly rising validation curve (≈0.47–0.49) would flatten each other on
    one shared scale, so each gets its own interval with 0.02 spacing."""
    fig = plt.figure(figsize=(12.8, 7.2))
    fig.subplots_adjust(left=.08, right=.98, top=.76, bottom=.16)
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 2.2], hspace=.10)
    ax_train = fig.add_subplot(grid[0])
    ax_val = fig.add_subplot(grid[1])
    x = curve["train_size"]
    xlim = (-x.max() * .06, x.max() * 1.06)

    ax_train.plot(x, curve["train_mean"], color=BLUE, marker="o", markersize=8, linewidth=2.5)
    ax_train.fill_between(x, curve["train_mean"] - curve["train_std"], curve["train_mean"] + curve["train_std"], color=BLUE, alpha=.14)
    for xi, tv in zip(x, curve["train_mean"]):
        ax_train.text(xi, tv + .006, f"{tv:.3f}", ha="center", fontsize=9, color=BLUE)
    ax_train.set_ylim(.595, .648); ax_train.set_yticks([.60, .62, .64])
    ax_train.set_xlim(*xlim); ax_train.set_xticks(x, [])
    ax_train.text(.012, .80, "Training folds", transform=ax_train.transAxes, fontsize=11, color=BLUE, weight="bold")

    ax_val.plot(x, curve["val_mean"], color=ORANGE, marker="o", markersize=8, linewidth=2.5)
    ax_val.fill_between(x, curve["val_mean"] - curve["val_std"], curve["val_mean"] + curve["val_std"], color=ORANGE, alpha=.14)
    for xi, vv in zip(x, curve["val_mean"]):
        ax_val.text(xi, vv + .008, f"{vv:.3f}", ha="center", fontsize=9, color=ORANGE)
    ax_val.set_ylim(.455, .52); ax_val.set_yticks([.46, .48, .50])
    ax_val.set_xlim(*xlim); ax_val.set_xticks(x, [f"{int(v):,}" for v in x])
    ax_val.text(.012, .80, "Validation folds", transform=ax_val.transAxes, fontsize=11, color=ORANGE, weight="bold")
    ax_val.set_xlabel("Training records", color=MUTED)

    for ax in (ax_train, ax_val):
        ax.grid(axis="y", color=GRID, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False); _style_axis(ax)
    ax_train.spines["bottom"].set_visible(False)
    ax_val.spines["top"].set_visible(False)
    gap = .012
    for ax in (ax_train, ax_val):
        ax.plot((-gap, +gap), (1 - gap, 1 + gap), transform=ax.transAxes, color=TEXT, clip_on=False, linewidth=1.2)
        ax.plot((-gap, +gap), (-gap, +gap), transform=ax.transAxes, color=TEXT, clip_on=False, linewidth=1.2)
    fig.text(.02, .46, "Macro-F1", rotation=90, va="center", color=MUTED, fontsize=11)
    fig.text(.08, .955, "Validation macro-F1 keeps rising with more training records", fontsize=20, color=TEXT, weight="bold", ha="left")
    fig.text(.08, .91, "Five training sizes × 3-fold cross-validation with the tuned parameters; bands show ±1 standard deviation", color=MUTED, fontsize=11, ha="left")
    fig.text(.08, .05, "The y-axis is split so both scales stay readable; learning curves describe model capacity and data volume, not future predictive performance.", fontsize=9, color=MUTED, ha="left")
    _save(fig, figure_dir / "04_learning_curve")


def plot_leakage_comparison(comparison: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.subplots_adjust(left=.08, right=.97, top=.76, bottom=.17)
    x = np.arange(len(comparison))
    width = .36
    bars_acc = ax.bar(x - width / 2, comparison["accuracy"], width, color=[GREY, BLUE, ORANGE], label="Accuracy")
    bars_f1 = ax.bar(x + width / 2, comparison["macro_f1"], width, color=[GREY, BLUE, ORANGE], alpha=.55, label="Macro-F1")
    for bars in (bars_acc, bars_f1):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .015, f"{bar.get_height():.3f}", ha="center", fontsize=11, color=TEXT)
    ax.text(2.18, 1.075, "Artificially inflated", ha="center", fontsize=12, color=ORANGE, weight="bold")
    ax.set_xticks(x, comparison["model"]); ax.set_ylabel("Holdout score", color=MUTED)
    ax.set_xlim(-.75, 3.15); ax.set_ylim(0, 1.16)
    ax.grid(axis="y", color=GRID, linewidth=.7)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); _style_axis(ax)
    fig.text(.08, .955, "Fields that encode the charge itself inflate accuracy", fontsize=20, color=TEXT, weight="bold", ha="left")
    fig.text(.08, .91, "Identical pipeline and parameters; the leakage model additionally sees LAW_CODE, PD_CD and OFNS_DESC", color=MUTED, fontsize=11, ha="left")
    fig.text(.08, .05, "The leakage-augmented model is a comparison for the demonstration only; the reported model excludes those fields.", fontsize=9, color=MUTED, ha="left")
    _save(fig, figure_dir / "05_leakage_comparison")


# ---------------------------------------------------------------- outputs


def write_outputs(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_results: pd.DataFrame = results["grid_results"]
    grid_results.to_csv(output_dir / "grid_search_results.csv", index=False)
    results["classification_report"].to_csv(output_dir / "classification_report.csv")
    confusion = pd.DataFrame(results["confusion_matrix"], index=[f"recorded_{c}" for c in SEVERITY_ORDER], columns=[f"predicted_{label}" for label in SEVERITY_LABELS.values()])
    confusion.to_csv(output_dir / "confusion_matrix.csv")
    results["feature_importance"].to_csv(output_dir / "feature_importance.csv", index=False)
    results["learning_curve"].to_csv(output_dir / "learning_curve.csv", index=False)
    results["leakage_comparison"].to_csv(output_dir / "leakage_comparison.csv", index=False)
    pdp_long = pd.concat(
        [frame.assign(feature={"AGE_GROUP": "Age group", "ARREST_BORO": "Borough", "MONTH": "Month"}[key])[["feature", "display", "p_felony"]].rename(columns={"display": "category"})
         for key, frame in results["pdp"].items()], ignore_index=True)
    pdp_long.to_csv(output_dir / "pdp_values.csv", index=False)
    best = {key.removeprefix("rf__"): (int(value) if isinstance(value, (int, np.integer)) else value) for key, value in results["grid"].best_params_.items()}
    best["best_cv_f1_macro"] = float(results["grid"].best_score_)
    (output_dir / "best_parameters.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    metrics = {key: value for key, value in results["metrics"].items() if key != "predictions"}
    metrics["majority_baseline_accuracy"] = results["baseline"]
    metrics["leakage_accuracy"] = results["leakage_metrics"]["accuracy"]
    metrics["leakage_macro_f1"] = results["leakage_metrics"]["macro_f1"]
    (output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata = {
        "records": int(results["n_records"]), "core_fmv_records": int(results["n_core"]),
        "train_size": int(results["n_train"]), "test_size": int(results["n_test"]),
        "random_state": RANDOM_STATE, "tuning_subsample_size": TUNING_SUBSAMPLE_SIZE,
        "scoring": "f1_macro", "test_size_share": TEST_SIZE,
        "features": list(CONTEXT_FEATURES), "leakage_columns_excluded": list(LEAKAGE_COLUMNS),
        "class_distribution": {SEVERITY_LABELS[code]: {"count": int(count), "share_pct": round(100 * count / results["n_core"], 4)}
                               for code, count in zip(SEVERITY_ORDER, results["class_counts"])},
        "best_parameters": best, "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"], "test_macro_f1": metrics["macro_f1"],
        "majority_baseline_accuracy": results["baseline"],
        "leakage_accuracy": results["leakage_metrics"]["accuracy"], "leakage_macro_f1": results["leakage_metrics"]["macro_f1"],
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_analysis(data_path: Path = DEFAULT_DATA, figure_dir: Path = DEFAULT_FIGURE_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    figure_dir.mkdir(parents=True, exist_ok=True); output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_data(data_path)
    split = split_core(frame)
    X_train = build_feature_frame(split["train"])
    X_test = build_feature_frame(split["test"])
    y_train, y_test = split["y_train"], split["y_test"]
    y_all = pd.concat([y_train, y_test])
    class_counts = [int((y_all == code).sum()) for code in SEVERITY_ORDER]
    baseline = majority_baseline(y_test)
    print("Tuning: 12 parameter combinations × stratified 5-fold CV on a 30,000-row subsample ...")
    grid, grid_results = tune_on_subsample(X_train, y_train)
    best_params = {key.removeprefix("rf__"): (int(value) if isinstance(value, (int, np.integer)) else value) for key, value in grid.best_params_.items()}
    print(f"Best parameters: {best_params}  (CV macro-F1 {grid.best_score_:.4f})")
    print("Fitting the tuned model on all 112,380 training records ...")
    final = make_pipeline(model_params=best_params, rf_n_jobs=-1)
    final.fit(X_train, y_train)
    metrics = evaluate_model(final, X_test, y_test)
    report = pd.DataFrame(classification_report(y_test, metrics["predictions"], labels=list(SEVERITY_ORDER), output_dict=True, zero_division=0)).T
    report.index = [SEVERITY_LABELS.get(str(name), str(name)) for name in report.index]
    report["support"] = report["support"].fillna(0).astype(int)
    print(f"Holdout: accuracy {metrics['accuracy']:.4f} · balanced {metrics['balanced_accuracy']:.4f} · macro-F1 {metrics['macro_f1']:.4f}  (majority baseline {baseline:.4f})")
    print("Learning curve: 5 training sizes × 3-fold CV ...")
    lc_pipe = make_pipeline(model_params=best_params, rf_n_jobs=1)
    curve = compute_learning_curve(lc_pipe, X_train, y_train)
    print("Partial dependence for age group, borough and month ...")
    pdp = compute_pdp(final, X_test)
    importance = compute_feature_importance(final)
    print("Leakage comparison model (charge-encoding fields added) ...")
    leakage_metrics = train_leakage_model(split["train"], split["test"], y_train, y_test, best_params)
    print(f"Leakage model: accuracy {leakage_metrics['accuracy']:.4f} vs contextual {metrics['accuracy']:.4f} (inflated by {leakage_metrics['accuracy'] - metrics['accuracy']:.4f})")
    majority_f1 = float(f1_score(y_test, ["M"] * len(y_test), average="macro", labels=list(SEVERITY_ORDER), zero_division=0))
    comparison = pd.DataFrame({
        "model": ["Majority baseline", "Contextual model", "Leakage-augmented model"],
        "accuracy": [baseline, metrics["accuracy"], leakage_metrics["accuracy"]],
        "macro_f1": [majority_f1, metrics["macro_f1"], leakage_metrics["macro_f1"]],
    })
    felony_share = float((y_test == "F").mean())
    results = {
        "frame": frame, "n_records": len(frame), "n_core": len(split["train"]) + len(split["test"]),
        "n_train": len(split["train"]), "n_test": len(split["test"]),
        "class_counts": class_counts, "baseline": baseline,
        "grid": grid, "grid_results": grid_results, "best_params": best_params,
        "metrics": metrics, "classification_report": report,
        "confusion_matrix": metrics["confusion_matrix"],
        "learning_curve": curve, "pdp": pdp, "feature_importance": importance,
        "leakage_metrics": leakage_metrics, "leakage_comparison": comparison, "felony_share": felony_share,
    }
    write_outputs(results, output_dir)
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), figure_dir)
    plot_feature_importance(importance, figure_dir)
    plot_partial_dependence(pdp, felony_share, figure_dir)
    plot_learning_curve(curve, figure_dir)
    plot_leakage_comparison(comparison, figure_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_analysis(args.data, args.figures, args.outputs)
    print(f"Part 3 modelling complete: {args.figures.resolve()} and {args.outputs.resolve()}")


if __name__ == "__main__":
    main()
