# ADR 0002: Patient-grouped cross-validation; everything learned inside the fold

- Status: accepted
- Date: 2026-11-24

## Context
Genomic classifiers often report optimistic accuracy because of leakage: the same patient in training
and test (TCGA has patients with more than one tumour sample), or features chosen on the whole data
set before splitting (for example "the 300 most informative genes"). Either inflates results in a
way that does not survive new data.

## Decision
- Folds are built with `StratifiedGroupKFold`, grouped by TCGA patient barcode and stratified by
  tumour type.
- Inside each training fold, and only there: the gene list, the recurrent protein changes, the
  feature scaling, and the regularisation strength C (inner 3-fold grouped CV on log loss).
- Out-of-fold probabilities are pooled for metrics, calibration and the abstention threshold.
- Baselines run on the same folds: class prior, and a model restricted to mutational-process
  features (TMB, SBS6, indel share). Gradient boosting on the same features is reported for comparison.
- Results are repeated over three seeds and reported as mean ± standard deviation.

## Consequences
- A test replaces the fitting function with a spy and fails if any training fold contains a
  held-out patient's tumour.
- Training is slower (nested CV), about 20 s for 1,870 tumours; acceptable for a research tool.
- Folds are not grouped by sequencing centre, so centre effects can still flatter the numbers; this
  is listed as a limit rather than hidden.
