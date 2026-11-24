# Changelog

## 1.0.0 — 2026-11-24
- MAF reader with row checks (required fields, alleles, variant types, duplicates, read-count and allele-frequency filters, pseudonymous sample IDs) and a per-reason rejection report.
- Per-tumour profiles: TMB (cBioPortal convention), SBS6, SBS96 from sequence context, genes and protein changes.
- Cohort statistics: gene recurrence, pairwise co-occurrence and mutual exclusivity (Fisher exact, Benjamini–Hochberg).
- Tumour-type classifier: multinomial logistic regression, feature space learned in-fold, balanced classes, abstention threshold from out-of-fold predictions.
- Evaluation: patient-grouped 5-fold CV with nested choice of C, baselines, gradient boosting comparison, calibration error, abstention curve.
- JSON model format scored with NumPy (no pickle).
- Pinned manifest of all 32 TCGA PanCancer Atlas studies with SHA-256-verified download.
- FastAPI service with viewer / analyst / admin keys, research console, in-memory uploads, audit trail, Prometheus metrics.
- Benchmark on ten public studies (docs/data/validation.md), synthetic demo, container and CI.
