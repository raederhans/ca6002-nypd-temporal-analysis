from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle


BG = "#FBFAF7"
INK = "#17253A"
MUTED = "#687586"
GRID = "#D9DEE5"
PALE = "#EEF1F4"
BURGUNDY = "#8B1118"
BURGUNDY_LIGHT = "#F1D9DA"
TEAL = "#147A8A"
TEAL_LIGHT = "#BFDDE1"
GOLD = "#D49A00"
GOLD_LIGHT = "#F5E5B8"
MAUVE = "#A85D86"
GREY = "#9BA5B1"
WHITE = "#FFFFFF"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Aptos", "DejaVu Sans"],
            "font.size": 15,
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.titlecolor": INK,
            "svg.fonttype": "none",
            "savefig.facecolor": BG,
        }
    )


def title_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.935, title, ha="left", va="top", fontsize=28, fontweight="bold", color=INK)
    fig.text(0.055, 0.885, subtitle, ha="left", va="top", fontsize=14.5, color=MUTED)
    fig.add_artist(Rectangle((0.055, 0.855), 0.052, 0.006, transform=fig.transFigure, color=BURGUNDY, lw=0))


def footer(fig: plt.Figure, extra: str = "") -> None:
    text = "Source: NYC Open Data / NYPD · dataset uip8-fykc · observed 1 Jan–30 Jun 2026 · retrieved 22 Aug 2026"
    if extra:
        text = f"{text} · {extra}"
    fig.text(0.055, 0.012, text, ha="left", va="bottom", fontsize=9.5, color=MUTED)


def finish(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches=None, pad_inches=0)
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches=None, pad_inches=0)
    svg_text = svg_path.read_text(encoding="utf-8")
    with svg_path.open("w", encoding="utf-8", newline="\n") as svg_file:
        svg_file.write("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n")
    plt.close(fig)


def strip_axes(ax: plt.Axes) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def iter_polygons(geometry: dict):
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geom_type == "Polygon":
        for ring in coordinates[:1]:
            yield ring
    elif geom_type == "MultiPolygon":
        for polygon in coordinates:
            if polygon:
                yield polygon[0]


def precinct_visual(data_root: Path, out_dir: Path) -> None:
    counts = pd.read_csv(data_root / "outputs" / "part2" / "precinct_counts.csv")
    count_map = dict(zip(counts["precinct"].astype(int), counts["arrest_records"].astype(int)))
    total = int(counts["arrest_records"].sum())
    top10 = counts.head(10).copy()
    top10_share = top10["arrest_records"].sum() / total * 100

    geo_path = data_root / "data" / "reference" / "nypd_police_precincts_simplified.geojson"
    geo = json.loads(geo_path.read_text(encoding="utf-8"))
    values = np.array(list(count_map.values()), dtype=float)
    norm = Normalize(vmin=values.min(), vmax=values.max())
    cmap = LinearSegmentedColormap.from_list("precinct", ["#E8F1F3", "#80B7C0", "#0D5F70"])

    fig = plt.figure(figsize=(16, 9))
    title_block(
        fig,
        "Arrests concentrate in a few precincts",
        f"Recorded arrest counts, not population-adjusted risk · n={total:,} records",
    )
    ax_map = fig.add_axes([0.055, 0.12, 0.46, 0.69])
    ax_rank = fig.add_axes([0.56, 0.17, 0.385, 0.60])

    focal_centroid = None
    for feature in geo["features"]:
        precinct = int(feature["properties"]["precinct"])
        value = count_map.get(precinct)
        face = PALE if value is None else cmap(norm(value))
        for ring in iter_polygons(feature["geometry"]):
            xy = np.asarray(ring, dtype=float)
            if len(xy) < 3:
                continue
            patch = Polygon(xy, closed=True, facecolor=face, edgecolor=WHITE, linewidth=0.55)
            ax_map.add_patch(patch)
            if precinct == 75:
                ax_map.add_patch(
                    Polygon(xy, closed=True, facecolor=BURGUNDY, edgecolor=WHITE, linewidth=1.2, zorder=6)
                )
                if focal_centroid is None:
                    focal_centroid = (float(xy[:, 0].mean()), float(xy[:, 1].mean()))

    ax_map.autoscale_view()
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.axis("off")
    ax_map.text(0.0, 1.01, "WHERE", transform=ax_map.transAxes, fontsize=11, color=MUTED, fontweight="bold")
    if focal_centroid:
        ax_map.annotate(
            "P75 · 5,341",
            xy=focal_centroid,
            xytext=(0.71, 0.42),
            textcoords=ax_map.transAxes,
            fontsize=13,
            fontweight="bold",
            color=BURGUNDY,
            arrowprops=dict(arrowstyle="-", color=BURGUNDY, lw=1.3),
            bbox=dict(boxstyle="round,pad=0.28", fc=BG, ec=BURGUNDY, lw=1),
        )
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.455, 0.18, 0.012, 0.31])
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, length=0, colors=MUTED)
    cb.set_label("records", fontsize=9, color=MUTED, labelpad=8)

    y = np.arange(len(top10))[::-1]
    vals = top10["arrest_records"].to_numpy()
    labels = [f"P{int(v)}" for v in top10["precinct"]]
    colors = [BURGUNDY] + [TEAL] * (len(top10) - 1)
    ax_rank.hlines(y, 0, vals, color=[BURGUNDY_LIGHT] + [TEAL_LIGHT] * 9, lw=7, zorder=1)
    ax_rank.scatter(vals, y, s=120, c=colors, marker="o", edgecolor=WHITE, linewidth=1.2, zorder=3)
    for yy, value in zip(y, vals):
        ax_rank.text(value + 105, yy, f"{value:,}", va="center", ha="left", fontsize=12.2, color=INK)
    ax_rank.set_yticks(y, labels)
    ax_rank.set_xlim(0, 5900)
    ax_rank.set_xticks([0, 2000, 4000, 6000])
    ax_rank.set_xticklabels(["0", "2k", "4k", "6k"])
    ax_rank.grid(axis="x", color=GRID, lw=0.8)
    ax_rank.set_axisbelow(True)
    strip_axes(ax_rank)
    ax_rank.text(0.0, 1.08, "TOP 10 PRECINCTS", transform=ax_rank.transAxes, fontsize=11, color=MUTED, fontweight="bold")
    ax_rank.text(
        0.0,
        -0.19,
        f"Top 10 = {top10_share:.1f}% of all records",
        transform=ax_rank.transAxes,
        fontsize=18,
        fontweight="bold",
        color=BURGUNDY,
    )
    ax_rank.text(
        0.0,
        -0.27,
        "Map answers where; the ranking answers how much.",
        transform=ax_rank.transAxes,
        fontsize=12.5,
        color=MUTED,
    )
    footer(fig, "precinct counts are raw workload observations")
    finish(fig, out_dir, "R05_precinct_map_top10")


def age_visual(data_root: Path, out_dir: Path) -> None:
    pct = pd.read_csv(data_root / "outputs" / "part2" / "age_severity_pct.csv")
    counts = pd.read_csv(data_root / "outputs" / "part2" / "age_severity_counts.csv")
    totals = counts[["Felony", "Misdemeanor", "Violation"]].sum(axis=1).astype(int)
    groups = pct["age_group"].tolist()
    y = np.arange(len(groups))[::-1]

    fig = plt.figure(figsize=(16, 9))
    title_block(
        fig,
        "Severity composition differs by age group",
        "Within-age-group shares of core Felony / Misdemeanor / Violation records",
    )
    ax = fig.add_axes([0.13, 0.25, 0.79, 0.49])
    ax.axhspan(y[0] - 0.48, y[0] + 0.48, color=BURGUNDY_LIGHT, alpha=0.45, zorder=0)

    left = np.zeros(len(pct))
    for column, color, hatch in [
        ("Felony", BURGUNDY, ""),
        ("Misdemeanor", GOLD, ""),
        ("Violation", MAUVE, "///"),
    ]:
        vals = pct[column].to_numpy()
        ax.barh(y, vals, left=left, height=0.58, color=color, edgecolor=BG, linewidth=1.0, hatch=hatch, label=column)
        left += vals

    for idx, yy in enumerate(y):
        f = pct.loc[idx, "Felony"]
        m = pct.loc[idx, "Misdemeanor"]
        v = pct.loc[idx, "Violation"]
        ax.text(f / 2, yy, f"{f:.1f}%", ha="center", va="center", fontsize=13, color=WHITE, fontweight="bold")
        ax.text(f + m / 2, yy, f"{m:.1f}%", ha="center", va="center", fontsize=13, color=INK, fontweight="bold")
        v_label = "V <0.1%" if 0 < v < 0.05 else f"V {v:.1f}%"
        ax.text(101.2, yy, v_label, ha="left", va="center", fontsize=11.5, color=MAUVE, fontweight="bold")
        ax.text(-3.2, yy, f"n={totals.iloc[idx]:,}", ha="right", va="center", fontsize=11.5, color=MUTED)

    ax.set_yticks(y, groups)
    ax.set_xlim(-18, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    strip_axes(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, 1.13),
        ncol=3,
        frameon=False,
        fontsize=12.5,
        handlelength=1.2,
        handletextpad=0.45,
    )
    fig.text(
        0.13,
        0.115,
        "Under 18 has the highest Felony share (63.2%), but ages 25–44 account for the most core records (80,645).",
        fontsize=15,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.13,
        0.077,
        "Composition is not age-specific crime risk; age-band widths are unequal.",
        fontsize=12.5,
        color=MUTED,
    )
    footer(fig, "core F/M/V denominator n=140,476")
    finish(fig, out_dir, "R06_age_severity_composition")


def tuning_visual(data_root: Path, out_dir: Path) -> None:
    df = pd.read_csv(data_root / "outputs" / "part3" / "grid_search_results.csv")
    df["depth_label"] = df["max_depth"].apply(lambda v: "No limit" if pd.isna(v) else str(int(v)))
    df["label"] = df.apply(
        lambda r: f"{int(r.n_estimators)} trees · depth {r.depth_label} · leaf {int(r.min_samples_leaf)}", axis=1
    )
    df = df.sort_values("mean_test_f1_macro", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    fig = plt.figure(figsize=(16, 9))
    title_block(
        fig,
        "Twelve tuning settings produce only small score differences",
        "Five-fold CV Macro F1 on a 30,000-record training subset · error bars show ±1 fold SD",
    )
    ax = fig.add_axes([0.30, 0.20, 0.62, 0.59])
    colors = np.where(df["n_estimators"].eq(200), TEAL, GOLD)
    markers = np.where(df["n_estimators"].eq(200), "o", "D")
    for i, row in df.iterrows():
        ax.errorbar(
            row["mean_test_f1_macro"],
            y[i],
            xerr=row["std_test_f1_macro"],
            fmt="none",
            ecolor=GRID,
            elinewidth=3,
            capsize=4,
            zorder=1,
        )
        ax.scatter(
            row["mean_test_f1_macro"],
            y[i],
            s=110,
            c=colors[i],
            marker=markers[i],
            edgecolor=WHITE,
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            row["mean_test_f1_macro"] + 0.0033,
            y[i],
            f"{row['mean_test_f1_macro']:.3f}",
            va="center",
            fontsize=11.5,
            color=INK,
        )
    best_idx = int(df["mean_test_f1_macro"].idxmax())
    best = df.loc[best_idx]
    ax.scatter(
        best["mean_test_f1_macro"],
        y[best_idx],
        s=260,
        facecolors="none",
        edgecolors=BURGUNDY,
        linewidth=2.3,
        zorder=4,
    )
    ax.annotate(
        "Selected · rank 1",
        xy=(best["mean_test_f1_macro"], y[best_idx]),
        xytext=(0.495, y[best_idx] - 1.1),
        fontsize=12.5,
        fontweight="bold",
        color=BURGUNDY,
        arrowprops=dict(arrowstyle="-", color=BURGUNDY, lw=1.3),
    )
    ax.set_yticks(y, df["label"])
    ax.set_xlim(0.425, 0.505)
    ax.set_xticks([0.44, 0.46, 0.48, 0.50])
    ax.set_xlabel("Mean CV Macro F1", labelpad=12, fontsize=12.5)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    strip_axes(ax)
    fig.text(
        0.30,
        0.105,
        "Best: 200 trees · no depth limit · min leaf 5 · Macro F1 0.4787",
        fontsize=15,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.30,
        0.066,
        "Test records were not used for model selection; overlapping uncertainty cautions against over-reading tiny gaps.",
        fontsize=12.2,
        color=MUTED,
    )
    footer(fig)
    finish(fig, out_dir, "R09_tuning_forest_plot")


def baseline_visual(data_root: Path, out_dir: Path) -> None:
    df = pd.read_csv(data_root / "outputs" / "part 4" / "evaluation_summary.csv")
    df = df[df["metric"].isin(["Accuracy", "Macro F1"])].copy()
    df["baseline"] = df["Majority Baseline"] * 100
    df["forest"] = df["Random Forest"] * 100
    df["delta"] = df["forest"] - df["baseline"]
    labels = df["metric"].tolist()
    y = np.array([1, 0])

    fig = plt.figure(figsize=(16, 9))
    title_block(
        fig,
        "Accuracy barely moves; Macro F1 doubles",
        "Majority-class baseline versus contextual Random Forest · held-out test set n=28,096",
    )
    ax = fig.add_axes([0.15, 0.27, 0.72, 0.43])
    for i, row in df.reset_index(drop=True).iterrows():
        yy = y[i]
        ax.plot([row["baseline"], row["forest"]], [yy, yy], color=GRID, lw=8, solid_capstyle="round", zorder=1)
        ax.scatter(row["baseline"], yy, s=180, facecolor=BG, edgecolor=GREY, linewidth=3, zorder=3)
        ax.scatter(row["forest"], yy, s=190, facecolor=BURGUNDY, edgecolor=WHITE, linewidth=1.5, zorder=4)
        ax.text(row["baseline"], yy + 0.18, f"Baseline {row['baseline']:.2f}%", ha="center", fontsize=12, color=MUTED)
        ax.text(row["forest"], yy - 0.22, f"Forest {row['forest']:.2f}%", ha="center", fontsize=12, color=BURGUNDY, fontweight="bold")
        ax.text(
            max(row["baseline"], row["forest"]) + 4.0,
            yy,
            f"+{row['delta']:.2f} pp",
            ha="left",
            va="center",
            fontsize=15,
            color=BURGUNDY if row["metric"] == "Macro F1" else MUTED,
            fontweight="bold",
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    strip_axes(ax)
    fig.text(
        0.15,
        0.095,
        "The near-overlap on accuracy is the point: context adds equal-class signal, not a strong headline classifier.",
        fontsize=14,
        color=INK,
        fontweight="bold",
    )
    footer(fig)
    finish(fig, out_dir, "R12_baseline_vs_forest")


def confusion_visual(data_root: Path, out_dir: Path) -> None:
    raw = pd.read_csv(data_root / "outputs" / "part3" / "confusion_matrix.csv", index_col=0)
    matrix = raw.to_numpy(dtype=int)
    row_pct = matrix / matrix.sum(axis=1, keepdims=True) * 100
    labels = ["Felony", "Misdemeanor", "Violation"]
    cmap = LinearSegmentedColormap.from_list("matrix", ["#EDF4F5", "#95C2C9", "#0B6575"])

    fig = plt.figure(figsize=(16, 9))
    title_block(
        fig,
        "The model mainly confuses Felony and Misdemeanor",
        "Rows = actual; columns = predicted · cells show within-row share and raw count · test set n=28,096",
    )
    ax = fig.add_axes([0.07, 0.15, 0.58, 0.58])
    ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=80, aspect="equal")
    for i in range(3):
        for j in range(3):
            value = row_pct[i, j]
            color = WHITE if value >= 50 else INK
            ax.text(
                j,
                i - 0.06,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=20,
                color=color,
                fontweight="bold",
            )
            ax.text(j, i + 0.22, f"({matrix[i, j]:,})", ha="center", va="center", fontsize=12.5, color=color)
    for i, j in [(0, 1), (1, 0)]:
        ax.add_patch(Rectangle((j - 0.49, i - 0.49), 0.98, 0.98, fill=False, edgecolor=BURGUNDY, lw=3.0))
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=12)
    ax.set_ylabel("Actual", fontsize=12.5, color=MUTED, labelpad=16, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color=BG, linestyle="-", linewidth=5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax_note = fig.add_axes([0.69, 0.20, 0.25, 0.54])
    ax_note.axis("off")
    ax_note.add_patch(
        FancyBboxPatch((0.0, 0.56), 1.0, 0.36, boxstyle="round,pad=0.018,rounding_size=0.025", fc=BURGUNDY_LIGHT, ec="none")
    )
    ax_note.text(0.06, 0.84, "THE DOMINANT ERROR", fontsize=10.5, color=BURGUNDY, fontweight="bold")
    ax_note.text(0.06, 0.71, "36.4%  F → M", fontsize=21, color=INK, fontweight="bold")
    ax_note.text(0.06, 0.61, "36.3%  M → F", fontsize=21, color=INK, fontweight="bold")
    ax_note.add_patch(
        FancyBboxPatch((0.0, 0.10), 1.0, 0.34, boxstyle="round,pad=0.018,rounding_size=0.025", fc=PALE, ec="none")
    )
    ax_note.text(0.06, 0.36, "READ THE DIAGONAL CAREFULLY", fontsize=10.5, color=MUTED, fontweight="bold")
    ax_note.text(0.06, 0.24, "Violation recall: 76.5%", fontsize=16.5, color=INK, fontweight="bold")
    ax_note.text(0.06, 0.14, "Recall does not account for\nfalse-positive predictions.", fontsize=12.5, color=MUTED, va="top")
    footer(fig)
    finish(fig, out_dir, "R13_confusion_matrix")


def class_performance_visual(data_root: Path, out_dir: Path) -> None:
    df = pd.read_csv(data_root / "outputs" / "part 4" / "class_performance.csv")
    y = np.array([2, 1, 0])
    precision = df["precision"].to_numpy() * 100
    recall = df["recall"].to_numpy() * 100
    f1 = df["f1"].to_numpy() * 100

    fig = plt.figure(figsize=(16, 9))
    title_block(
        fig,
        "Minority-class recall comes with low precision",
        "Per-class metrics on the held-out test set · support shown beside each actual class",
    )
    ax = fig.add_axes([0.17, 0.25, 0.66, 0.45])
    ax.axhspan(-0.38, 0.38, color=BURGUNDY_LIGHT, alpha=0.60, zorder=0)
    for i, yy in enumerate(y):
        lo, hi = sorted([precision[i], recall[i]])
        ax.plot([lo, hi], [yy, yy], color=GRID, lw=8, solid_capstyle="round", zorder=1)
        ax.scatter(precision[i], yy, s=170, facecolor=TEAL, edgecolor=WHITE, linewidth=1.4, marker="o", zorder=3)
        ax.scatter(recall[i], yy, s=180, facecolor=GOLD, edgecolor=WHITE, linewidth=1.4, marker="s", zorder=3)
        ax.scatter(f1[i], yy, s=165, facecolor=MAUVE, edgecolor=WHITE, linewidth=1.4, marker="D", zorder=4)
        positions = [(precision[i], -0.19, TEAL, "P"), (recall[i], 0.20, GOLD, "R"), (f1[i], -0.19, MAUVE, "F1")]
        for value, dy, color, short in positions:
            x = value
            ha = "center"
            if i == 0 and short == "P":
                x = value - 1.2
                ha = "right"
            elif i == 0 and short == "F1":
                x = value + 1.2
                ha = "left"
            ax.text(x, yy + dy, f"{short} {value:.1f}%", ha=ha, va="center", fontsize=10.8, color=color, fontweight="bold")
    ylabels = [f"{row['class']}   n={int(row['support']):,}" for _, row in df.iterrows()]
    ax.set_yticks(y, ylabels)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    strip_axes(ax)
    fig.text(
        0.17,
        0.105,
        "Violation: 1,315 false positives among 1,605 predictions; only 290 were correct.",
        fontsize=15,
        color=BURGUNDY,
        fontweight="bold",
    )
    fig.text(
        0.17,
        0.067,
        "High recall alone does not make minority-class predictions reliable.",
        fontsize=12.5,
        color=MUTED,
    )
    footer(fig)
    finish(fig, out_dir, "R14_class_precision_recall")


def leakage_visual(data_root: Path, out_dir: Path) -> None:
    df = pd.read_csv(data_root / "outputs" / "part3" / "leakage_comparison.csv")
    context = float(df.loc[df["model"].eq("Contextual model"), "accuracy"].iloc[0] * 100)
    leakage = float(df.loc[df["model"].eq("Leakage-augmented model"), "accuracy"].iloc[0] * 100)

    fig = plt.figure(figsize=(16, 9))
    title_block(
        fig,
        "Charge-related fields change the prediction task",
        "Accuracy comparison using the same data split · the near-perfect result is a leakage diagnostic",
    )
    ax = fig.add_axes([0.16, 0.18, 0.52, 0.58])
    bars = ax.bar(
        [0, 1],
        [context, leakage],
        width=0.52,
        color=[TEAL, BURGUNDY_LIGHT],
        edgecolor=[TEAL, BURGUNDY],
        linewidth=[0, 2.2],
    )
    bars[1].set_hatch("///")
    for x, value, color in [(0, context, TEAL), (1, leakage, BURGUNDY)]:
        ax.text(x, value + 2.2, f"{value:.2f}%", ha="center", va="bottom", fontsize=22, color=color, fontweight="bold")
    ax.set_xticks([0, 1], ["Context only", "Charge-related fields"])
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    strip_axes(ax)
    ax.text(1, 8, "LEAKAGE-PRONE\nDIAGNOSTIC", ha="center", va="bottom", fontsize=11.5, color=BURGUNDY, fontweight="bold")

    ax_note = fig.add_axes([0.72, 0.25, 0.23, 0.45])
    ax_note.axis("off")
    ax_note.add_patch(
        FancyBboxPatch((0.0, 0.50), 1.0, 0.45, boxstyle="round,pad=0.025,rounding_size=0.03", fc=BURGUNDY_LIGHT, ec="none")
    )
    ax_note.text(0.07, 0.86, "WHAT CHANGED", fontsize=10.5, color=BURGUNDY, fontweight="bold")
    ax_note.text(0.07, 0.73, "LAW_CODE", fontsize=17, color=INK, fontweight="bold")
    ax_note.text(0.07, 0.63, "PD_CD", fontsize=17, color=INK, fontweight="bold")
    ax_note.text(0.07, 0.53, "OFNS_DESC", fontsize=17, color=INK, fontweight="bold")
    ax_note.text(
        0.02,
        0.34,
        "These fields encode or closely\ndescribe the recorded charge.",
        fontsize=13,
        color=INK,
        va="top",
    )
    ax_note.text(
        0.02,
        0.13,
        "Keep the contextual result as\nthe answer to the original question.",
        fontsize=12.5,
        color=MUTED,
        va="top",
    )
    footer(fig)
    finish(fig, out_dir, "R15_charge_field_leakage")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standalone redesigned CA6002 chart assets.")
    parser.add_argument("--data-root", type=Path, required=True, help="Repository snapshot containing outputs/part1-4.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for PNG and SVG assets.")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    out_dir = args.out.resolve()
    configure_style()
    precinct_visual(data_root, out_dir)
    age_visual(data_root, out_dir)
    tuning_visual(data_root, out_dir)
    baseline_visual(data_root, out_dir)
    confusion_visual(data_root, out_dir)
    class_performance_visual(data_root, out_dir)
    leakage_visual(data_root, out_dir)
    print(f"Wrote redesigned chart assets to {out_dir}")


if __name__ == "__main__":
    main()
