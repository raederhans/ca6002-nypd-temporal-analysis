"""Build and execute the reader-facing Part 2 notebook."""
from __future__ import annotations

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_spatial_demographic_analysis.ipynb"


def build_notebook(output_path: Path = NOTEBOOK_PATH) -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Part 2 — Spatial and demographic analysis

This notebook reproduces the Person 2 analysis using the unchanged Part 1 cleaned baseline. The unit of analysis is one recorded arrest. These data describe enforcement and recording activity, not underlying crime incidence."""),
        nbf.v4.new_code_cell("""from pathlib import Path
import sys
import pandas as pd
from IPython.display import display, Image, Markdown

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from analyze_spatial_demographic import load_data, calculate_outputs, quality_tables

DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'nypd_arrests_clean.csv'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'part2'
FIGURE_DIR = PROJECT_ROOT / 'figures' / 'part2'
frame = load_data(DATA_PATH)
results = calculate_outputs(frame)
len(frame)"""),
        nbf.v4.new_markdown_cell("## 1. Field and quality confirmation"),
        nbf.v4.new_code_cell("""field_quality, validation = quality_tables(frame)
display(field_quality)
display(validation)"""),
        nbf.v4.new_markdown_cell("""The shared dataset retains all six fields required by Part 2. Core F/M/V severity analysis uses 140,476 records; missing and non-core categories remain visible in the audit."""),
        nbf.v4.new_markdown_cell("## 2. Spatial concentration"),
        nbf.v4.new_code_cell("""display(results['borough_counts'])
display(results['precinct_counts'].head(15))"""),
        nbf.v4.new_code_cell("display(Image(filename=str(FIGURE_DIR / '01_precinct_arrest_map.png'), width=1000))"),
        nbf.v4.new_code_cell("display(Image(filename=str(FIGURE_DIR / '02_top_precincts.png'), width=1000))"),
        nbf.v4.new_markdown_cell("""Spatial patterns are reported as recorded arrest counts. They are not adjusted for population, exposure, footfall, or precinct area and therefore are not crime rates."""),
        nbf.v4.new_markdown_cell("## 3. Age group × severity"),
        nbf.v4.new_code_cell("""display(results['age_severity_counts'])
display(results['age_severity_pct'].round(2))
print("Cramér's V:", round(results['age_cramers_v'], 3))"""),
        nbf.v4.new_code_cell("display(Image(filename=str(FIGURE_DIR / '03_age_severity_profile.png'), width=1000))"),
        nbf.v4.new_markdown_cell("""The under-18 category has the highest felony share. This is a descriptive association within recorded arrests and does not demonstrate that age causes arrest severity."""),
        nbf.v4.new_markdown_cell("## 4. Borough × severity"),
        nbf.v4.new_code_cell("""display(results['borough_severity_pct'].round(2))
display(results['borough_severity_delta_pp'].round(2))
print("Cramér's V:", round(results['borough_cramers_v'], 3))"""),
        nbf.v4.new_code_cell("display(Image(filename=str(FIGURE_DIR / '04_borough_severity_deviation.png'), width=1000))"),
        nbf.v4.new_markdown_cell("""Borough composition differences are modest. Row-normalised shares separate severity composition from the unequal number of records in each borough."""),
        nbf.v4.new_markdown_cell("""## 5. Interpretation boundary

- Arrest records reflect policing, reporting, enforcement, and administrative processes.
- Raw geographic counts are not population-adjusted rates.
- The six-month snapshot is not evidence of a long-term trend.
- Associations do not establish causal relationships.
- F/M/V composition excludes 1,394 missing/non-core records; that denominator is stated in all relevant outputs."""),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT)}})
    executed = client.execute()
    nbf.write(executed, output_path)
    return output_path


if __name__ == "__main__":
    print(build_notebook())
