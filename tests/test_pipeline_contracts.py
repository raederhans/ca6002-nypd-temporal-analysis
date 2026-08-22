from __future__ import annotations

import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Barrier

import nbformat
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_temporal import calculate_monthly_arrests  # noqa: E402
from build_notebook import build_notebook  # noqa: E402
from download_nypd_data import (  # noqa: E402
    _download_once,
    _new_part_path,
    _promote_snapshot_no_clobber,
    download_snapshot,
)
from validate_part1 import (  # noqa: E402
    _duplicate_evidence_matches,
    _expected_clean_source,
    _expected_duplicate_evidence,
)


class PipelineContractTests(unittest.TestCase):
    def test_frozen_same_date_snapshot_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True)
            frozen = raw_dir / (
                f"nypd_arrests_ytd_{datetime.now().astimezone().date().isoformat()}.csv"
            )
            original = b"arrest_key,arrest_date\n1,01/01/2026\n"
            frozen.write_bytes(original)

            with self.assertRaisesRegex(FileExistsError, "will not be overwritten"):
                download_snapshot(root)

            self.assertEqual(frozen.read_bytes(), original)

    def test_atomic_promotion_does_not_clobber_concurrently_created_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            part = root / "snapshot.csv.part"
            frozen = root / "snapshot.csv"
            part.write_bytes(b"downloaded-candidate")
            frozen.write_bytes(b"concurrent-owner")

            with self.assertRaisesRegex(FileExistsError, "was not overwritten"):
                _promote_snapshot_no_clobber(part, frozen)

            self.assertEqual(frozen.read_bytes(), b"concurrent-owner")
            self.assertEqual(part.read_bytes(), b"downloaded-candidate")

    def test_concurrent_downloaders_use_distinct_parts_and_cannot_mutate_winner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frozen = root / "snapshot.csv"
            parts = [_new_part_path(frozen), _new_part_path(frozen)]
            self.assertNotEqual(parts[0], parts[1])
            parts[0].write_bytes(b"candidate-a")
            parts[1].write_bytes(b"candidate-b")
            barrier = Barrier(2)

            def promote(part: Path) -> bool:
                barrier.wait()
                try:
                    _promote_snapshot_no_clobber(part, frozen)
                    return True
                except FileExistsError:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(promote, parts))

            self.assertEqual(results.count(True), 1)
            published = frozen.read_bytes()
            losing_part = parts[results.index(False)]
            losing_part.write_bytes(b"late-write-from-loser")
            self.assertEqual(frozen.read_bytes(), published)

    def test_pagination_uses_unique_tie_breaker_across_duplicate_key_boundary(self):
        class FakeResponse:
            def __init__(self, text: str) -> None:
                self.text = text

            def raise_for_status(self) -> None:
                return None

        class FakeSession:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def get(self, _url: str, *, params, timeout):
                self.calls.append(dict(params))
                self.assert_contract(params)
                header = (
                    "arrest_key,arrest_date,pd_desc,__socrata_row_id\n"
                )
                if int(params["$offset"]) == 0:
                    return FakeResponse(
                        header
                        + "1,2026-01-01T00:00:00.000,A,row-ngii-yf5a~8r9a\n"
                        + "1,2026-01-02T00:00:00.000,B,row-zbff~mqbh_7h5c\n"
                    )
                return FakeResponse(
                    header + "1,2026-01-03T00:00:00.000,C,row-vzzp-opaque\n"
                )

            @staticmethod
            def assert_contract(params) -> None:
                if params["$order"] != "arrest_key, :id":
                    raise AssertionError(params["$order"])
                if params["$select"] != "*, :id as __socrata_row_id":
                    raise AssertionError(params["$select"])

        with tempfile.TemporaryDirectory() as temp_dir:
            part = Path(temp_dir) / "page.csv.part"
            session = FakeSession()

            stats = _download_once(
                session,
                part,
                expected_rows=3,
                order_field="arrest_key",
                page_size=2,
            )

            output = pd.read_csv(part, dtype="string")
            self.assertEqual(output["arrest_key"].tolist(), ["1", "1", "1"])
            self.assertNotIn("__socrata_row_id", output.columns)
            self.assertEqual(stats["duplicate_arrest_key_rows"], 2)
            self.assertTrue(stats["pagination_tie_breaker_unique"])
            self.assertTrue(stats["pagination_order_is_total"])
            self.assertEqual(len(session.calls), 2)

    def test_notebook_routes_effective_missingness_audit_to_analysis(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "contract.ipynb"
            build_notebook(project_root=PROJECT_ROOT, output_path=destination)
            notebook = nbformat.read(destination, as_version=4)
            code = "\n".join(
                cell.source for cell in notebook.cells if cell.cell_type == "code"
            )

            self.assertIn(
                'audit_csv=OUTPUT_DIR / "missingness_summary.csv"', code
            )
            self.assertNotIn(
                'audit_csv=OUTPUT_DIR / "schema_summary.csv"', code
            )

    def test_monthly_aggregation_marks_both_truncated_boundary_months(self):
        dates = pd.date_range("2026-01-10", "2026-02-15", freq="D")
        daily = pd.DataFrame(
            {
                "date": dates,
                "arrest_count": 1,
                "rolling_7d_mean": 1.0,
                "is_zero_count_day": False,
            }
        )

        monthly = calculate_monthly_arrests(daily)

        self.assertEqual(monthly["is_partial_month"].tolist(), [True, True])
        self.assertEqual(
            monthly["partial_reason"].tolist(),
            ["starts 2026-01-10", "ends 2026-02-15"],
        )
        self.assertEqual(monthly["calendar_days_in_scope"].tolist(), [22, 15])

    def test_validator_baseline_removes_only_exact_duplicates_and_retains_dirty_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "snapshot.csv"
            raw_path.write_text(
                "arrest_key,arrest_date,pd_cd,ky_cd,arrest_precinct,"
                "jurisdiction_code,pd_desc\n"
                "1,01/01/2026,001,101,010,0, A \n"
                "1,01/01/2026,001,101,010,0, A \n"
                "1,not-a-date,002,102,011,0,B\n",
                encoding="utf-8",
            )

            baseline, stats = _expected_clean_source(raw_path)

            self.assertEqual(len(baseline), 2)
            self.assertEqual(stats["exact_duplicate_rows_removed"], 1)
            self.assertEqual(
                stats["non_identical_duplicated_arrest_keys_retained"], 1
            )
            self.assertEqual(
                stats["rows_with_non_identical_duplicated_arrest_keys_retained"],
                2,
            )
            self.assertEqual(stats["invalid_date_values_retained_as_nat"], 1)
            self.assertTrue(pd.isna(baseline.loc[1, "ARREST_DATE"]))
            self.assertEqual(baseline.loc[0, "PD_CD"], "001")

    def test_duplicate_evidence_rejects_same_cardinality_tampering(self):
        raw = pd.DataFrame(
            {
                "arrest_key": ["1", "1"],
                "arrest_date": ["01/01/2026", "01/02/2026"],
                "pd_desc": ["A", "B"],
            }
        )
        expected = _expected_duplicate_evidence(raw)
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_path = Path(temp_dir) / "evidence.csv"
            tampered = expected.copy()
            tampered["PD_DESC"] = ["X", "Y"]
            tampered.to_csv(evidence_path, index=False)

            self.assertFalse(_duplicate_evidence_matches(raw, evidence_path))

            expected.to_csv(evidence_path, index=False)
            self.assertTrue(_duplicate_evidence_matches(raw, evidence_path))

    def test_duplicate_evidence_distinguishes_blank_from_na_aliases(self):
        raw = pd.DataFrame(
            {
                "arrest_key": ["1", "1"],
                "arrest_date": ["01/01/2026", "01/02/2026"],
                "pd_desc": [pd.NA, pd.NA],
            }
        )
        expected = _expected_duplicate_evidence(raw)
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_path = Path(temp_dir) / "evidence.csv"
            for alias in ("NA", "NULL"):
                tampered = expected.copy()
                tampered.loc[0, "PD_DESC"] = alias
                tampered.to_csv(evidence_path, index=False)
                self.assertFalse(
                    _duplicate_evidence_matches(raw, evidence_path), alias
                )

            expected.to_csv(evidence_path, index=False)
            self.assertTrue(_duplicate_evidence_matches(raw, evidence_path))


if __name__ == "__main__":
    unittest.main()
