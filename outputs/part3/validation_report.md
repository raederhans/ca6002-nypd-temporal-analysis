# Part 3 Validation Report

Overall assessment: **Ready for review**

- PASS — processed row count is 141,870
- PASS — core F/M/V denominator is 140,476 and 1,394 rows are excluded
- PASS — stratified 80/20 split is 112,380 / 28,096
- PASS — split reconciles to the core denominator
- PASS — random state 42 is fixed
- PASS — all ten data artifacts and five handoff documents exist
- PASS — holdout metrics are in range and the leakage model inflates accuracy by more than 0.05
- PASS — confusion matrix row sums reconcile to 28,096
- PASS — classification report matches the per-class metrics JSON
- PASS — grid search contains exactly 12 parameter combinations
- PASS — rank-1 grid parameters equal the stored best parameters
- PASS — feature importance aggregates back to eight inputs summing to 100%
- PASS — learning curve has five sizes and a training/validation gap at full size
- PASS — five figure families exist as 2560×1440 PNG and valid SVG
- PASS — Part 3 notebook executed top-to-bottom without errors
- PASS — handoff language avoids unsupported crime/causal claims
- PASS — key documents use the recorded-arrests framing
- PASS — slide plan names the artificial leakage inflation
- PASS — slide plan contains four Part 3 slides
