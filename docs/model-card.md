# Model card: tumour type from somatic mutations

## Summary
Multinomial logistic regression that estimates which of ten TCGA tumour types a tumour's somatic
mutations most resemble, with calibrated probabilities and an "uncertain" answer below a confidence
threshold. Research use only.

## Intended use
- Exploring public or ethically approved research cohorts: quality checks, cohort context, checking
  whether a sample's mutations look like its recorded type (for example to flag possible sample
  swaps or mislabelled entries for human review).
- Teaching and method development.

## Out of scope
- Any decision about a patient: diagnosis, cancer of unknown primary work-up, prognosis, treatment
  or trial eligibility.
- Tumour types outside the ten it was trained on: the model will still return one of the ten.
- Targeted-panel data, metastases, non-TCGA pipelines, without re-validation.

## Training data
TCGA PanCancer Atlas (MC3 calls) via the cBioPortal datahub, commit `026a269c69`: ACC 91, DLBC 37,
GBM 395, KICH 65, LGG 510, MESO 82, OV 409, TGCT 144, UCS 57, UVM 80 tumours (1,870 tumours,
1,863 patients). Labels are study membership. Mostly primary tumours from adult patients in
the United States; ancestry is skewed toward European ancestry as in TCGA overall.

## Features
log₂(1 + TMB), indel share, six substitution-class shares (standardised); up to 300 genes hit by
protein-changing mutations and the protein changes seen in ≥ 3 training tumours (binary). All
learned on training data only. The 96 trinucleotide channels are available but off by default
(they reduced every metric on this data).

## Evaluation
Patient-grouped 5-fold cross-validation, C chosen by inner CV, three seeds:
accuracy 0.745 ± 0.004, balanced accuracy 0.670 ± 0.014, macro-F1 0.671 ± 0.010, top-3 0.917,
log loss 0.867, expected calibration error 0.027. With abstention below p = 0.80 the model answers
49 % of tumours and is right on 91 % of those. Per-type results and the confusion matrix are in
[validation.md](data/validation.md).

## Known weaknesses
- GBM and LGG are confused (34 and 71 tumours respectively); IDH1 status drives the split.
- UCS is often predicted as OV.
- Low-mutation types (ACC, KICH, MESO, TGCT) are recognised in only 40–74 % of cases.
- Calibration was measured on pooled out-of-fold predictions within TCGA; it may not hold elsewhere.

## Ethical considerations
- The data is de-identified and public, released for research under TCGA and cBioPortal terms.
- Errors could mislead research conclusions about individual samples; the "uncertain" answer and
  per-sample explanations exist to make weak evidence visible.
- Performance may differ across ancestries; this was not measured.

## File format
`model.json`: `format = oncoai-model/1`, classes, threshold, feature space, coefficients,
intercepts, and `meta` (versions, data source, cross-validated metrics, abstention, cohort summary).
Loaded with shape, finiteness and size checks; never unpickled.
