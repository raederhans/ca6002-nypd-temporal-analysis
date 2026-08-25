"""Part 2 spatial and demographic analysis of the shared cleaned NYPD data.

Run from any directory with:
    python src/analyze_spatial_demographic.py

The module reads the Part 1 processed baseline, writes reproducible summary
tables, and exports four presentation-ready figures in PNG and SVG formats.
Arrest records are treated as recorded enforcement activity, not crime rates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "nypd_arrests_clean.csv"
DEFAULT_BOUNDARIES = PROJECT_ROOT / "data" / "reference" / "nypd_police_precincts_simplified.geojson"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figures" / "part2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part2"

REQUIRED_COLUMNS = (
    "ARREST_BORO", "ARREST_PRECINCT", "AGE_GROUP", "LAW_CAT_CD",
    "LATITUDE", "LONGITUDE",
)
BOROUGH_LABELS = {"B": "Bronx", "K": "Brooklyn", "M": "Manhattan", "Q": "Queens", "S": "Staten Island"}
SEVERITY_LABELS = {"F": "Felony", "M": "Misdemeanor", "V": "Violation"}
AGE_ORDER = ["<18", "18-24", "25-44", "45-64", "65+"]
SEVERITY_ORDER = ["Felony", "Misdemeanor", "Violation"]
SEVERITY_COLORS = {"Felony": "#0072B2", "Misdemeanor": "#E69F00", "Violation": "#CC79A7"}
TEXT = "#1F2937"
MUTED = "#4B5563"
GRID = "#D9DEE3"
BLUE = "#0072B2"
ORANGE = "#E69F00"


def load_data(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in frame]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    frame = frame.copy()
    frame["BOROUGH"] = frame["ARREST_BORO"].astype("string").str.strip().str.upper().map(BOROUGH_LABELS)
    frame["SEVERITY"] = frame["LAW_CAT_CD"].astype("string").str.strip().str.upper().map(SEVERITY_LABELS)
    frame["AGE_STD"] = frame["AGE_GROUP"].astype("string").str.strip()
    for column in ("ARREST_PRECINCT", "LATITUDE", "LONGITUDE"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def cramers_v(table: pd.DataFrame) -> float:
    observed = table.to_numpy(dtype=float)
    if observed.size == 0 or observed.sum() == 0:
        return float("nan")
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    chi_square = np.divide((observed - expected) ** 2, expected, out=np.zeros_like(expected), where=expected > 0).sum()
    denominator = observed.sum() * min(observed.shape[0] - 1, observed.shape[1] - 1)
    return math.sqrt(chi_square / denominator) if denominator > 0 else float("nan")


def calculate_outputs(frame: pd.DataFrame) -> dict[str, object]:
    valid_borough = frame[frame["BOROUGH"].isin(BOROUGH_LABELS.values())]
    core = frame[frame["SEVERITY"].isin(SEVERITY_ORDER)]
    borough_counts = valid_borough["BOROUGH"].value_counts().rename_axis("borough").reset_index(name="arrest_records")
    precinct_counts = frame.loc[frame["ARREST_PRECINCT"].between(1, 123), "ARREST_PRECINCT"].value_counts().rename_axis("precinct").reset_index(name="arrest_records")
    precinct_counts["precinct"] = precinct_counts["precinct"].astype(int)
    precinct_counts["share_pct"] = 100 * precinct_counts["arrest_records"] / len(frame)
    borough_severity = pd.crosstab(core["BOROUGH"], core["SEVERITY"]).reindex(columns=SEVERITY_ORDER, fill_value=0)
    borough_severity.index.name = "borough"
    borough_pct = borough_severity.div(borough_severity.sum(axis=1), axis=0) * 100
    citywide_pct = borough_severity.sum(axis=0) / borough_severity.to_numpy().sum() * 100
    borough_delta = borough_pct.subtract(citywide_pct, axis=1)
    age_severity = pd.crosstab(core["AGE_STD"], core["SEVERITY"]).reindex(AGE_ORDER, fill_value=0).reindex(columns=SEVERITY_ORDER, fill_value=0)
    age_severity.index.name = "age_group"
    age_pct = age_severity.div(age_severity.sum(axis=1), axis=0) * 100
    return {
        "borough_counts": borough_counts,
        "precinct_counts": precinct_counts,
        "borough_severity_counts": borough_severity,
        "borough_severity_pct": borough_pct,
        "borough_severity_delta_pp": borough_delta,
        "age_severity_counts": age_severity,
        "age_severity_pct": age_pct,
        "citywide_severity_pct": citywide_pct,
        "borough_cramers_v": cramers_v(borough_severity),
        "age_cramers_v": cramers_v(age_severity),
    }


def quality_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = []
    for column in REQUIRED_COLUMNS:
        series = frame[column]
        fields.append({
            "field": column,
            "non_null": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(100 * series.isna().mean(), 4),
            "unique_values": int(series.nunique(dropna=True)),
        })
    checks = [
        ("row_count", len(frame)),
        ("recognised_borough_pct", 100 * frame["BOROUGH"].isin(BOROUGH_LABELS.values()).mean()),
        ("recognised_age_pct", 100 * frame["AGE_STD"].isin(AGE_ORDER).mean()),
        ("core_fmv_severity_pct", 100 * frame["SEVERITY"].isin(SEVERITY_ORDER).mean()),
        ("valid_precinct_pct", 100 * frame["ARREST_PRECINCT"].between(1, 123).mean()),
        ("valid_nyc_coordinate_pct", 100 * (frame["LATITUDE"].between(40.45, 40.95) & frame["LONGITUDE"].between(-74.30, -73.65)).mean()),
        ("exact_duplicate_rows_pct", 100 * frame.duplicated().mean()),
    ]
    validation = pd.DataFrame([{"check": key, "value": round(float(value), 4)} for key, value in checks])
    return pd.DataFrame(fields), validation


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.tick_params(colors=MUTED, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def _iter_rings(geometry: dict) -> Iterable[list[list[float]]]:
    if geometry["type"] == "Polygon":
        for ring in geometry["coordinates"]:
            yield ring
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            for ring in polygon:
                yield ring


def plot_precinct_map(precinct_counts: pd.DataFrame, boundary_path: Path, figure_dir: Path) -> None:
    geo = json.loads(boundary_path.read_text(encoding="utf-8"))
    counts = precinct_counts.set_index("precinct")["arrest_records"].to_dict()
    values = [counts.get(int(f["properties"]["precinct"]), 0) for f in geo["features"]]
    cmap = LinearSegmentedColormap.from_list("nypd_blue", ["#EFF6FA", "#7DB8D1", "#005A83"])
    norm = Normalize(vmin=min(values), vmax=max(values))
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.subplots_adjust(left=.04, right=.89, top=.84, bottom=.10)
    for feature in geo["features"]:
        precinct = int(feature["properties"]["precinct"])
        for ring in _iter_rings(feature["geometry"]):
            ax.add_patch(Polygon(ring, closed=True, facecolor=cmap(norm(counts.get(precinct, 0))), edgecolor="white", linewidth=.45))
    all_xy = [point for f in geo["features"] for ring in _iter_rings(f["geometry"]) for point in ring]
    xs, ys = zip(*all_xy)
    ax.set_xlim(min(xs) - .01, max(xs) + .01); ax.set_ylim(min(ys) - .01, max(ys) + .01)
    ax.set_aspect("equal"); ax.axis("off")
    fig.text(.05, .94, "Recorded arrests are concentrated in a subset of NYPD precincts", fontsize=20, color=TEXT, weight="bold", ha="left")
    fig.text(.05, .905, "141,870 records, 1 Jan–30 Jun 2026 · raw counts, not population-adjusted rates", color=MUTED, fontsize=11, ha="left")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); cbar = fig.colorbar(sm, ax=ax, fraction=.025, pad=.015)
    cbar.set_label("Recorded arrests", color=MUTED); cbar.ax.tick_params(labelsize=9, colors=MUTED)
    fig.text(.05, .035, "Source: cleaned NYPD Arrest Data YTD; NYC DCP police-precinct boundaries. Arrest records ≠ crime incidence.", fontsize=9, color=MUTED, ha="left")
    _save(fig, figure_dir / "01_precinct_arrest_map")


def plot_precinct_ranking(precinct_counts: pd.DataFrame, figure_dir: Path) -> None:
    data = precinct_counts.head(15).sort_values("arrest_records")
    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    y = np.arange(len(data)); ax.hlines(y, 0, data["arrest_records"], color=GRID, linewidth=2)
    ax.scatter(data["arrest_records"], y, s=70, color=ORANGE, zorder=3)
    for yi, value in zip(y, data["arrest_records"]): ax.text(value + 70, yi, f"{value:,}", va="center", fontsize=10, color=TEXT)
    ax.set_yticks(y, [f"Precinct {p}" for p in data["precinct"]]); ax.set_xlabel("Recorded arrests", color=MUTED)
    ax.set_title("The 75th Precinct records the highest arrest volume", loc="left", fontsize=20, color=TEXT, weight="bold", pad=12)
    top10_share = 100 * precinct_counts.head(10)["arrest_records"].sum() / precinct_counts["arrest_records"].sum()
    ax.text(0, 1.01, f"Top 10 precincts account for {top10_share:.1f}% of all records", transform=ax.transAxes, color=MUTED, fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=.7); ax.spines[["top", "right", "left"]].set_visible(False); _style_axis(ax)
    ax.text(0, -.12, "Raw counts are descriptive and are not adjusted for population, exposure, or precinct area.", transform=ax.transAxes, fontsize=9, color=MUTED)
    _save(fig, figure_dir / "02_top_precincts")


def plot_age_severity(age_counts: pd.DataFrame, age_pct: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    x = np.arange(len(AGE_ORDER))
    for severity in SEVERITY_ORDER:
        ax.plot(x, age_pct[severity], marker="o", markersize=8, linewidth=2.5, label=severity, color=SEVERITY_COLORS[severity])
        for xi, value in zip(x, age_pct[severity]): ax.text(xi, value + (1.5 if severity != "Violation" else 1), f"{value:.1f}%", ha="center", fontsize=9, color=SEVERITY_COLORS[severity])
    ax.set_xticks(x, AGE_ORDER); ax.set_ylim(0, 70); ax.set_ylabel("Share within age group (%)", color=MUTED); ax.set_xlabel("Age group", color=MUTED)
    ax.set_title("Under-18 arrest records have a distinctly higher felony share", loc="left", fontsize=20, color=TEXT, weight="bold", pad=12)
    ax.text(0, 1.01, "Severity composition within each age group; 140,476 core F/M/V records", transform=ax.transAxes, color=MUTED, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=.7); ax.legend(frameon=False, ncol=3, loc="upper right"); ax.spines[["top", "right"]].set_visible(False); _style_axis(ax)
    ax.text(0, -.14, "Descriptive association only: age does not establish a cause of recorded arrest severity.", transform=ax.transAxes, fontsize=9, color=MUTED)
    _save(fig, figure_dir / "03_age_severity_profile")


def plot_borough_delta(delta: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    matrix = delta.reindex(columns=SEVERITY_ORDER)
    limit = float(np.abs(matrix.to_numpy()).max())
    image = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]; ax.text(j, i, f"{value:+.2f} pp", ha="center", va="center", color="white" if abs(value) > 1.6 else TEXT, fontsize=11, weight="bold")
    ax.set_xticks(range(len(SEVERITY_ORDER)), SEVERITY_ORDER, fontsize=11); ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=11)
    ax.tick_params(axis="x", pad=12)
    ax.set_title("Borough severity profiles differ only modestly from the citywide mix", loc="left", fontsize=20, color=TEXT, weight="bold", pad=12)
    ax.text(0, 1.01, "Percentage-point difference from citywide severity shares", transform=ax.transAxes, color=MUTED, fontsize=11)
    cbar = fig.colorbar(image, ax=ax, fraction=.025, pad=.03); cbar.set_label("Percentage-point difference", color=MUTED); cbar.ax.tick_params(colors=MUTED)
    ax.tick_params(length=0); [spine.set_visible(False) for spine in ax.spines.values()]
    ax.text(0, -.10, "Row-normalised composition controls for different borough sample sizes; it does not estimate causal effects.", transform=ax.transAxes, fontsize=9, color=MUTED)
    _save(fig, figure_dir / "04_borough_severity_deviation")


def write_outputs(frame: pd.DataFrame, results: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields, validation = quality_tables(frame)
    fields.to_csv(output_dir / "field_quality.csv", index=False); validation.to_csv(output_dir / "validation_checks.csv", index=False)
    for key in ("borough_counts", "precinct_counts"):
        results[key].to_csv(output_dir / f"{key}.csv", index=False)
    for key in ("borough_severity_counts", "borough_severity_pct", "borough_severity_delta_pp", "age_severity_counts", "age_severity_pct"):
        results[key].to_csv(output_dir / f"{key}.csv")
    metadata = {
        "records": len(frame), "date_start": "2026-01-01", "date_end": "2026-06-30",
        "core_fmv_records": int(frame["SEVERITY"].notna().sum()),
        "non_core_or_missing_severity": int(frame["SEVERITY"].isna().sum()),
        "borough_cramers_v": round(float(results["borough_cramers_v"]), 4),
        "age_cramers_v": round(float(results["age_cramers_v"]), 4),
        "top_10_precinct_share_pct": round(100 * results["precinct_counts"].head(10)["arrest_records"].sum() / len(frame), 2),
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_analysis(data_path: Path = DEFAULT_DATA, boundary_path: Path = DEFAULT_BOUNDARIES, figure_dir: Path = DEFAULT_FIGURE_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    figure_dir.mkdir(parents=True, exist_ok=True); output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_data(data_path); results = calculate_outputs(frame); write_outputs(frame, results, output_dir)
    plot_precinct_map(results["precinct_counts"], boundary_path, figure_dir)
    plot_precinct_ranking(results["precinct_counts"], figure_dir)
    plot_age_severity(results["age_severity_counts"], results["age_severity_pct"], figure_dir)
    plot_borough_delta(results["borough_severity_delta_pp"], figure_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data", type=Path, default=DEFAULT_DATA); parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES); parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURE_DIR); parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(); run_analysis(args.data, args.boundaries, args.figures, args.outputs)
    print(f"Part 2 analysis complete: {args.figures.resolve()} and {args.outputs.resolve()}")


if __name__ == "__main__":
    main()
