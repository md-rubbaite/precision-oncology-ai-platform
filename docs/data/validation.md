# Validation on public data

All numbers on this page come from two commands on the data fetched by `oncoai fetch --studies benchmark`:

```bash
oncoai train --report report.json --boosting   # sections 3–5 (seed 7)
python examples/benchmark.py                   # sections 1, 2, 6, 7 (seeds 7, 11, 23)
```

They were produced with Python 3.13, NumPy 2.5, scikit-learn 1.9 on 24 November 2026.

## 1. Data

| | |
|---|---|
| Source | cBioPortal datahub, commit `026a269c6946f41815481af3263d2539db91fbb2` (2026-11-20), TCGA PanCancer Atlas 2018 studies |
| Mutation calls | MC3 consensus calls on whole-exome data ([Ellrott et al., 2018](https://doi.org/10.1016/j.cels.2018.03.002)) |
| Integrity | Every file checked against the SHA-256 and size in `src/oncoai/studies.toml` (taken from the datahub's Git LFS pointers) |
| Studies | ACC, DLBC, GBM, KICH, LGG, MESO, OV, TGCT, UCS, UVM |
| Tumours / patients | 1,870 / 1,863 (seven patients have two tumour samples; folds are grouped by patient) |
| Label | The study the tumour belongs to |

**Why ten studies.** When this was built, the datahub's LFS host served the files of these ten
studies and returned HTTP 404 for the other 22 from the build environment. The selection is
therefore an availability accident, not a design choice: it includes two brain tumour types that
are hard to separate (GBM, LGG) and several small, low-mutation types, and it excludes high-TMB
types (lung, melanoma, colorectal, endometrial) where mutational signatures carry strong signal.
All 32 studies are in the manifest; `oncoai fetch --studies all` and `oncoai train --studies all`
run the same pipeline on them.

## 2. TMB agreement with cBioPortal

cBioPortal publishes `TMB_NONSYNONYMOUS` per sample: nonsynonymous mutations divided by 30 Mb.

| Check | Result |
|---|---|
| Identical to cBioPortal | 66.9 % of 1,870 tumours |
| Within two mutations (0.07 per Mb) | 97.4 % |
| Largest difference | 53 mutations (one tumour) |
| Pearson r on log(1 + TMB) | 0.9999 |

Counting `Splice_Region` as protein-changing gave the closest match (236 of 395 GBM tumours
identical, against 132 without it). The remaining small differences come from rules in cBioPortal's
own pipeline that the MAF does not expose (this platform also drops exact duplicate rows).

## 3. Cross-validation scheme

- 5 folds, stratified by tumour type, **grouped by patient**: no patient is in both training and test.
- Inside each training fold: gene list (genes hit in ≥ 3 % of one type's tumours, at most 300),
  recurrent protein changes (≥ 3 tumours), feature scaling, and C (inner 3-fold grouped CV on log
  loss over 0.1, 0.3, 1, 3). C = 1 was chosen in every fold.
- Class weights are balanced, so small types are not ignored.
- Out-of-fold probabilities from all five folds are pooled for the tables below.

## 4. Overall results

Three seeds (7, 11, 23), mean ± standard deviation:

| Method | Accuracy | Balanced acc. | Macro-F1 | Top-3 | Log loss | ECE |
|---|---|---|---|---|---|---|
| **Logistic regression** | **0.745 ± 0.004** | **0.670 ± 0.014** | **0.671 ± 0.010** | **0.917 ± 0.004** | **0.867 ± 0.003** | **0.027 ± 0.005** |
| Mutational process only | 0.486 ± 0.005 | 0.452 ± 0.005 | 0.390 ± 0.004 | 0.779 ± 0.003 | 1.669 ± 0.002 | 0.099 ± 0.004 |
| Class prior | 0.273 | 0.100 | 0.043 | 0.703 | 1.933 | 0.000* |

Seed 7 only, same features and folds:

| Method | Accuracy | Balanced acc. | Macro-F1 | Top-3 | Log loss | ECE |
|---|---|---|---|---|---|---|
| Logistic regression | 0.741 | 0.656 | 0.663 | 0.921 | 0.863 | 0.028 |
| Gradient boosting (200 iterations, balanced) | 0.742 | 0.622 | 0.646 | 0.919 | 1.525 | 0.189 |

\* A constant prior is perfectly calibrated and useless; ECE alone is not a quality measure.

Boosting reaches the same accuracy but is over-confident (log loss 1.53, ECE 0.19). The served
model is the logistic regression (ADR 0003).

## 5. Per tumour type (logistic regression, seed 7)

| Type | Tumours | Recall | Precision | F1 | Boosting recall |
|---|---|---|---|---|---|
| ACC | 91 | 0.40 | 0.40 | 0.40 | 0.32 |
| DLBC | 37 | 0.59 | 0.92 | 0.72 | 0.54 |
| GBM | 395 | 0.75 | 0.71 | 0.73 | 0.79 |
| KICH | 65 | 0.52 | 0.33 | 0.41 | 0.37 |
| LGG | 510 | 0.79 | 0.91 | 0.85 | 0.81 |
| MESO | 82 | 0.44 | 0.43 | 0.43 | 0.46 |
| OV | 409 | 0.84 | 0.83 | 0.84 | 0.85 |
| TGCT | 144 | 0.74 | 0.59 | 0.66 | 0.74 |
| UCS | 57 | 0.58 | 0.77 | 0.66 | 0.42 |
| UVM | 80 | 0.90 | 0.97 | 0.94 | 0.91 |

Confusion matrix (rows: true type, columns: predicted):

| true \ predicted | ACC | DLBC | GBM | KICH | LGG | MESO | OV | TGCT | UCS | UVM |
|---|---|---|---|---|---|---|---|---|---|---|
| **ACC** | **36** | · | 5 | 9 | · | 11 | 13 | 17 | · | · |
| **DLBC** | 2 | **22** | 9 | 1 | · | 2 | · | · | 1 | · |
| **GBM** | 15 | 1 | **296** | 14 | 34 | 6 | 21 | 6 | 2 | · |
| **KICH** | 5 | · | 7 | **34** | · | 4 | 5 | 9 | 1 | · |
| **LGG** | 6 | 1 | 71 | 8 | **404** | 5 | 5 | 10 | · | · |
| **MESO** | 11 | · | 6 | 9 | · | **36** | 5 | 14 | 1 | · |
| **OV** | 7 | · | 13 | 15 | 3 | 8 | **345** | 13 | 5 | · |
| **TGCT** | 8 | · | 1 | 12 | 1 | 11 | 2 | **107** | · | 2 |
| **UCS** | 1 | · | 4 | · | · | · | 19 | · | **33** | · |
| **UVM** | · | · | 2 | 1 | · | 1 | · | 4 | · | **72** |

Reading it:
- **LGG → GBM (71) and GBM → LGG (34).** Both are diffuse gliomas. IDH1 mutations point to LGG; tumours without them look like GBM whatever their label.
- **UCS → OV (19).** Both carry TP53 mutations in most tumours and have similar mutational processes.
- **ACC, KICH, MESO, TGCT.** Low mutation counts (median TMB 0.4–0.9 per Mb) and few recurrent drivers, so there is little to separate them.
- **UVM.** GNAQ or GNA11 hotspot mutations in almost every tumour; recognised in 90 %.

## 6. Abstention

Out-of-fold confidence threshold vs. coverage (seed 7):

| Abstain below | Answers | Accuracy on answered |
|---|---|---|
| 0.0 (never) | 100 % | 74.1 % |
| 0.5 | 78.6 % | 83.3 % |
| 0.6 | 69.7 % | 86.1 % |
| 0.7 | 59.7 % | 89.3 % |
| **0.8 (served)** | **48.6 %** | **91.4 %** |
| 0.9 | 33.1 % | 94.0 % |

The served threshold is the lowest at which out-of-fold accuracy on answered tumours reaches 90 %.
Half of the tumours get "uncertain". That is the honest answer for exome data on these types.

The console screenshot uses a model trained without eight patients (two each of LGG, OV, TGCT, UVM;
its own threshold is 0.75). The four OV and UVM tumours were answered correctly. The four LGG and
TGCT tumours came back "uncertain"; their most likely labels (GBM, MESO, KICH) would all have been
wrong.

## 7. Ablation: 96 trinucleotide channels

| Features | Accuracy | Balanced acc. | Macro-F1 | Log loss | ECE |
|---|---|---|---|---|---|
| Without SBS96 (served) | 0.741 | 0.656 | 0.663 | 0.863 | 0.028 |
| With SBS96 | 0.684 | 0.578 | 0.579 | 1.367 | 0.132 |

82 % of these tumours have fewer than 100 single-base substitutions (median 52), so 96 fractions
are mostly noise and the model over-fits them. SBS96 stays available (`oncoai train --sbs96`) for cohorts
with higher mutation counts.

## 8. Gene pairs against published patterns

Fisher's exact test with Benjamini–Hochberg correction (`oncoai cohort`):

| Cohort | Genes | Tendency | Odds ratio | q | Published pattern |
|---|---|---|---|---|---|
| LGG | IDH1, TP53 | co-occurrence | 8.2 | 3 × 10⁻¹⁷ | IDH-mutant astrocytoma carries TP53 and ATRX |
| LGG | TP53, ATRX | co-occurrence | 41.9 | 3 × 10⁻⁵⁸ | as above |
| LGG | CIC, FUBP1 | co-occurrence | 14.3 | 8 × 10⁻¹⁶ | IDH-mutant 1p/19q-codeleted oligodendroglioma carries CIC and FUBP1 |
| LGG | TP53, CIC | mutual exclusivity | 0.04 | 6 × 10⁻²⁷ | the two subtypes rarely overlap |
| LGG | ATRX, CIC | mutual exclusivity | 0.04 | 1 × 10⁻¹⁹ | as above |
| UVM | GNAQ, GNA11 | mutual exclusivity | 0.01 | 5 × 10⁻¹³ | alternative activating mutations in the same pathway |
| UVM | BAP1, SF3B1 | mutual exclusivity (n.s.) | 0.25 | 0.56 | reported as near-exclusive; 80 tumours are too few to show it here |

Published patterns: TCGA Research Network, NEJM 2015 (lower-grade glioma) and Robertson et al.,
Cancer Cell 2017 (uveal melanoma). The test reproduces the strong patterns and, correctly, does not
claim the weak one.

## 9. What these numbers do not show

- Performance on tumours outside TCGA, on metastases, on targeted panels or on other variant callers.
- Performance on the 22 tumour types not in the benchmark.
- Robustness to batch or sequencing-centre effects within TCGA (folds are not split by centre).
- Anything about clinical usefulness.
