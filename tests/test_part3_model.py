from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from sklearn.model_selection import GridSearchCV, learning_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from train_predictive_model import (
    CONTEXT_FEATURES, F1_MACRO, LEAKAGE_COLUMNS, RANDOM_STATE, SEVERITY_ORDER,
    build_feature_frame, check_no_leakage, compute_feature_importance, evaluate_model,
    load_data, majority_baseline, make_pipeline, split_core,
)


class Part3ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_data(PROJECT_ROOT / "data" / "processed" / "nypd_arrests_clean.csv")
        cls.core = cls.frame[cls.frame["LAW_CAT_CD"].astype("string").str.strip().str.upper().isin(SEVERITY_ORDER)].copy()
        cls.y_core = cls.core["LAW_CAT_CD"].astype("string").str.strip().str.upper()
        cls.sample = cls.core.sample(2000, random_state=RANDOM_STATE)
        cls.y_sample = cls.sample["LAW_CAT_CD"].astype("string").str.strip().str.upper()
        cls.X_sample = build_feature_frame(cls.sample)

    def test_leakage_columns_rejected(self):
        for name in LEAKAGE_COLUMNS:
            with self.assertRaises(ValueError):
                check_no_leakage([name])
        check_no_leakage(CONTEXT_FEATURES)  # must not raise

    def test_context_feature_frame_shape_and_dtypes(self):
        frame = build_feature_frame(self.sample)
        self.assertEqual(list(frame.columns), list(CONTEXT_FEATURES))
        for column in CONTEXT_FEATURES[:6]:
            self.assertTrue(pd.api.types.is_string_dtype(frame[column]))
        leaky = build_feature_frame(self.sample, include_leakage=True)
        self.assertEqual(len(leaky.columns), 11)

    def test_split_stratification(self):
        split = split_core(self.frame)
        full_share = self.y_core.value_counts(normalize=True)
        test_share = split["y_test"].value_counts(normalize=True)
        for code in SEVERITY_ORDER:
            self.assertLess(abs(test_share[code] - full_share[code]), 0.005)

    def test_majority_baseline(self):
        self.assertLess(abs(majority_baseline(self.y_core) - 0.5903), 0.002)

    def test_smoke_train_evaluate(self):
        pipe = make_pipeline(model_params={"n_estimators": 20, "max_depth": 10})
        pipe.fit(self.X_sample, self.y_sample)
        metrics = evaluate_model(pipe, self.X_sample, self.y_sample)
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "per_class"):
            self.assertIn(key, metrics)
        self.assertEqual(len(metrics["per_class"]), 3)
        self.assertTrue(0 < metrics["macro_f1"] < 1)
        self.assertEqual(np.array(metrics["confusion_matrix"]).shape, (3, 3))

    def test_determinism_same_seed(self):
        results = []
        for _ in range(2):
            pipe = make_pipeline(model_params={"n_estimators": 20, "max_depth": 10})
            pipe.fit(self.X_sample, self.y_sample)
            results.append(evaluate_model(pipe, self.X_sample, self.y_sample)["macro_f1"])
        self.assertEqual(results[0], results[1])

    def test_pdp_categorical_smoke(self):
        pipe = make_pipeline(model_params={"n_estimators": 20, "max_depth": 10})
        pipe.fit(self.X_sample, self.y_sample)
        result = partial_dependence(
            pipe, self.X_sample, features=["AGE_GROUP"], categorical_features=["AGE_GROUP"],
            response_method="predict_proba",
        )
        self.assertEqual(result["average"].shape, (3, 5))
        self.assertTrue(np.all((result["average"] >= 0) & (result["average"] <= 1)))

    def test_grid_search_smoke(self):
        X, y = self.X_sample.iloc[:800], self.y_sample.iloc[:800]
        grid = GridSearchCV(
            make_pipeline(), {"rf__n_estimators": [10, 20]},
            scoring=F1_MACRO, cv=2, n_jobs=1,
        )
        grid.fit(X, y)
        self.assertEqual(len(grid.cv_results_["params"]), 2)
        self.assertIsInstance(grid.best_index_, (int, np.integer))

    def test_importance_aggregation_sums_to_one(self):
        pipe = make_pipeline(model_params={"n_estimators": 20, "max_depth": 10})
        pipe.fit(self.X_sample, self.y_sample)
        importance = compute_feature_importance(pipe)
        self.assertAlmostEqual(importance["importance"].sum(), 1.0, places=9)
        self.assertEqual(len(importance), 8)

    def test_learning_curve_smoke(self):
        X, y = self.X_sample.iloc[:800], self.y_sample.iloc[:800]
        sizes, train_scores, val_scores = learning_curve(
            make_pipeline(model_params={"n_estimators": 20, "max_depth": 10}), X, y,
            train_sizes=[0.5, 1.0], cv=2,
            scoring=F1_MACRO, n_jobs=1,
        )
        self.assertEqual(sizes.shape, (2,))
        self.assertEqual(train_scores.shape, (2, 2))
        self.assertEqual(val_scores.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
