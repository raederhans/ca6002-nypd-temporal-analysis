from __future__ import annotations

import sys
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from analyze_spatial_demographic import calculate_outputs, cramers_v, load_data


class SpatialDemographicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_data(PROJECT_ROOT / "data" / "processed" / "nypd_arrests_clean.csv")
        cls.results = calculate_outputs(cls.frame)

    def test_shared_baseline_is_not_filtered(self):
        self.assertEqual(len(self.frame), 141870)
        self.assertEqual(int(self.results["precinct_counts"]["arrest_records"].sum()), 141870)

    def test_core_severity_denominator_is_explicit(self):
        core = int(self.results["age_severity_counts"].to_numpy().sum())
        self.assertEqual(core, 140476)
        self.assertEqual(len(self.frame) - core, 1394)

    def test_row_normalised_age_percentages_sum_to_100(self):
        sums = self.results["age_severity_pct"].sum(axis=1)
        self.assertTrue(((sums - 100).abs() < 1e-9).all())

    def test_borough_delta_uses_citywide_weighted_share(self):
        counts = self.results["borough_severity_counts"]
        citywide = counts.sum(axis=0) / counts.to_numpy().sum() * 100
        reconstructed = self.results["borough_severity_pct"].subtract(citywide, axis=1)
        pd.testing.assert_frame_equal(reconstructed, self.results["borough_severity_delta_pp"])

    def test_cramers_v_zero_for_independent_table(self):
        table = pd.DataFrame([[10, 20], [20, 40]])
        self.assertAlmostEqual(cramers_v(table), 0.0)

    def test_expected_key_findings(self):
        precinct = self.results["precinct_counts"].iloc[0]
        self.assertEqual(int(precinct["precinct"]), 75)
        self.assertEqual(int(precinct["arrest_records"]), 5341)
        self.assertAlmostEqual(self.results["age_severity_pct"].loc["<18", "Felony"], 63.2116, places=3)
        self.assertLess(self.results["borough_cramers_v"], 0.1)
        self.assertLess(self.results["age_cramers_v"], 0.1)


if __name__ == "__main__":
    unittest.main()
