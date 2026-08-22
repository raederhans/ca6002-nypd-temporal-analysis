from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clean_nypd_data import (  # noqa: E402
    audit_dataset,
    build_duplicate_key_evidence,
    build_missingness_summary,
    clean_dataset,
    load_raw_data,
    resolve_raw_path,
    standardize_column_names,
)


class CleanNypdDataTests(unittest.TestCase):
    def test_cleaning_removes_only_exact_copies_and_retains_other_key_versions(self):
        original = pd.DataFrame(
            {
                "arrest_key": ["1", "1", "1", "2"],
                "arrest_date": [
                    "01/01/2026",
                    "01/01/2026",
                    "not-a-date",
                    None,
                ],
                "law_cat_cd": [" F ", " F ", "M", "V"],
                "latitude": [" 40.70 ", " 40.70 ", "bad", "40.71"],
                "longitude": ["-73.90", "-73.90", "-73.91", "-73.92"],
            }
        )

        cleaned, stats = clean_dataset(original)

        self.assertEqual(stats["original_rows"], 4)
        self.assertEqual(stats["exact_duplicate_rows_removed"], 1)
        self.assertEqual(stats["final_processed_rows"], 3)
        self.assertEqual(stats["non_identical_duplicated_arrest_keys_retained"], 1)
        self.assertEqual(
            stats["rows_with_non_identical_duplicated_arrest_keys_retained"], 2
        )
        self.assertEqual(stats["invalid_date_values_retained_as_nat"], 1)
        self.assertEqual(stats["missing_date_values_retained_as_nat"], 1)
        self.assertEqual(cleaned.loc[cleaned.index[0], "LAW_CAT_CD"], "F")
        self.assertTrue(pd.isna(cleaned.loc[cleaned.index[1], "ARREST_DATE"]))
        self.assertTrue(pd.isna(cleaned.loc[cleaned.index[1], "LATITUDE"]))
        self.assertEqual(cleaned.loc[cleaned.index[0], "DAY_OF_WEEK"], "Thursday")
        self.assertEqual(cleaned.loc[cleaned.index[0], "DAY_OF_WEEK_NUM"], 4)

    def test_duplicate_evidence_distinguishes_exact_and_non_exact_rows(self):
        frame = pd.DataFrame(
            {
                "arrest_key": [1, 1, 1, 2],
                "arrest_date": ["01/01/2026", "01/01/2026", "01/02/2026", "x"],
            }
        )

        evidence = build_duplicate_key_evidence(frame)

        self.assertEqual(len(evidence), 3)
        self.assertEqual(evidence["AUDIT_ARREST_KEY"].nunique(), 1)
        self.assertFalse(evidence["AUDIT_ALL_ROWS_IDENTICAL_FOR_KEY"].any())
        self.assertEqual(
            int(evidence["AUDIT_ROW_PART_OF_EXACT_DUPLICATE"].sum()), 2
        )

    def test_identifier_codes_become_nullable_text_without_float_suffixes(self):
        frame = pd.DataFrame(
            {
                "arrest_key": [318207860.0, None, "0007"],
                "pd_cd": [439.0, None, "00439"],
                "ky_cd": [109.0, None, "00109"],
                "arrest_precinct": [61.0, None, "061"],
                "jurisdiction_code": [0.0, None, "00"],
            }
        )

        cleaned, _ = clean_dataset(frame)

        expected_first = {
            "ARREST_KEY": "318207860",
            "PD_CD": "439",
            "KY_CD": "109",
            "ARREST_PRECINCT": "61",
            "JURISDICTION_CODE": "0",
        }
        expected_leading_zero = {
            "ARREST_KEY": "0007",
            "PD_CD": "00439",
            "KY_CD": "00109",
            "ARREST_PRECINCT": "061",
            "JURISDICTION_CODE": "00",
        }
        for column in expected_first:
            self.assertEqual(str(cleaned[column].dtype), "string")
            self.assertEqual(cleaned.loc[0, column], expected_first[column])
            self.assertTrue(pd.isna(cleaned.loc[1, column]))
            self.assertEqual(cleaned.loc[2, column], expected_leading_zero[column])

    def test_code_preserving_csv_load_keeps_original_spelling_and_missingness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = root / "nypd_arrests_ytd_2026-08-22.csv"
            snapshot.write_text(
                "arrest_key,pd_cd,ky_cd,arrest_precinct,jurisdiction_code\n"
                "0007,00439,00109,061,00\n"
                ",,,,\n",
                encoding="utf-8",
            )

            loaded = load_raw_data(
                snapshot,
                project_root=root,
                preserve_code_text=True,
            )

            self.assertEqual(loaded.loc[0, "arrest_key"], "0007")
            self.assertEqual(loaded.loc[0, "pd_cd"], "00439")
            self.assertTrue(pd.isna(loaded.loc[1, "arrest_key"]))
            self.assertTrue(pd.isna(loaded.loc[1, "pd_cd"]))

    def test_law_category_validation_separates_core_noncore_and_sentinel(self):
        frame = pd.DataFrame(
            {"law_cat_cd": ["F", "M", "V", "I", "9", "(null)", None]}
        )

        report = audit_dataset(frame, snapshot_date="2026-08-22")
        law = report["categorical_validation"]["columns"]["LAW_CAT_CD"]

        self.assertEqual(law["core_severity_count"], 3)
        self.assertEqual(law["known_non_core_value_counts"], {"I": 1})
        self.assertEqual(law["source_missing_sentinel_counts"], {"(null)": 1})
        self.assertEqual(law["unrecognised_non_core_value_counts"], {"9": 1})
        self.assertEqual(law["missing_count"], 1)

        missingness = build_missingness_summary(frame).set_index("column_name")
        self.assertEqual(missingness.loc["law_cat_cd", "missing_count"], 1)
        self.assertEqual(missingness.loc["law_cat_cd", "sentinel_missing_count"], 1)
        self.assertEqual(
            missingness.loc["law_cat_cd", "effective_missing_count"], 2
        )

    def test_latest_snapshot_discovery_uses_filename_date(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True)
            older = raw_dir / "nypd_arrests_ytd_2026-08-01.csv"
            newer = raw_dir / "nypd_arrests_ytd_2026-08-22.csv"
            unrelated = raw_dir / "other.csv"
            older.write_text("a\n1\n", encoding="utf-8")
            newer.write_text("a\n2\n", encoding="utf-8")
            unrelated.write_text("a\n3\n", encoding="utf-8")

            self.assertEqual(resolve_raw_path(project_root=root), newer.resolve())

    def test_standardisation_fails_on_column_collision(self):
        frame = pd.DataFrame(columns=["arrest key", "ARREST_KEY"])
        with self.assertRaisesRegex(ValueError, "collisions"):
            standardize_column_names(frame)


if __name__ == "__main__":
    unittest.main()
