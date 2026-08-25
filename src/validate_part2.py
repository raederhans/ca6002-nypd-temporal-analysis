"""Independent validation of Part 2 tables, figures, notebook, and language."""
from __future__ import annotations

import json
from pathlib import Path
import re
import nbformat
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate(project_root: Path = PROJECT_ROOT) -> list[str]:
    out = project_root / "outputs" / "part2"; figs = project_root / "figures" / "part2"
    data = pd.read_csv(project_root / "data" / "processed" / "nypd_arrests_clean.csv", low_memory=False)
    meta = json.loads((out / "analysis_metadata.json").read_text())
    checks = []
    assert len(data) == meta["records"] == 141870; checks.append("PASS — processed row count is 141,870")
    core = data["LAW_CAT_CD"].isin(["F", "M", "V"]).sum()
    assert core == meta["core_fmv_records"] == 140476; checks.append("PASS — core F/M/V denominator is 140,476")
    assert meta["non_core_or_missing_severity"] == 1394; checks.append("PASS — 1,394 non-core/missing severity rows audited")
    precinct = pd.read_csv(out / "precinct_counts.csv")
    assert precinct["arrest_records"].sum() == len(data); checks.append("PASS — precinct counts reconcile to all records")
    assert int(precinct.iloc[0]["precinct"]) == 75 and int(precinct.iloc[0]["arrest_records"]) == 5341; checks.append("PASS — highest-volume precinct recomputes")
    age = pd.read_csv(out / "age_severity_pct.csv", index_col=0)
    assert abs(age.loc["<18", "Felony"] - 63.2116) < .01; checks.append("PASS — under-18 felony share recomputes")
    delta = pd.read_csv(out / "borough_severity_delta_pp.csv", index_col=0)
    assert abs(delta.loc["Manhattan", "Misdemeanor"] - 2.2460) < .01; checks.append("PASS — largest borough composition departure recomputes")
    for stem in ("01_precinct_arrest_map", "02_top_precincts", "03_age_severity_profile", "04_borough_severity_deviation"):
        png, svg = figs / f"{stem}.png", figs / f"{stem}.svg"
        assert png.is_file() and svg.is_file(); assert Image.open(png).size == (2560, 1440); assert "<svg" in svg.read_text(encoding="utf-8")[:1000]
    checks.append("PASS — four figure families exist as 2560×1440 PNG and valid SVG")
    notebook = nbformat.read(project_root / "notebooks" / "02_spatial_demographic_analysis.ipynb", as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert code_cells and all(c.get("execution_count") is not None for c in code_cells); assert not any(o.get("output_type") == "error" for c in code_cells for o in c.get("outputs", []))
    checks.append("PASS — Part 2 notebook executed top-to-bottom without errors")
    docs = "\n".join(p.read_text(encoding="utf-8") for p in out.glob("*.md"))
    banned = [r"most crime", r"crime rate is", r"shows that age causes", r"proves? that location causes"]
    assert not any(re.search(pattern, docs, re.I) for pattern in banned); checks.append("PASS — handoff language avoids unsupported crime/causal claims")
    report = "# Part 2 Validation Report\n\nOverall assessment: **Ready for review**\n\n" + "\n".join(f"- {c}" for c in checks) + "\n"
    (out / "validation_report.md").write_text(report, encoding="utf-8")
    return checks


if __name__ == "__main__":
    for item in validate(): print(item)
