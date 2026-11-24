# ADR 0003: Serve a calibrated linear model that can say "uncertain"

- Status: accepted
- Date: 2026-11-24

## Context
On the benchmark, gradient boosting and logistic regression reach the same accuracy (0.742 vs 0.741,
seed 7). Boosting is badly calibrated: expected calibration error 0.19 and log loss 1.53, against
0.03 and 0.86 for logistic regression. A researcher who reads "p = 0.9" should be able to trust that
about nine in ten such calls are right. Exome data on these tumour types also does not support a
confident answer for every tumour.

## Decision
- Serve multinomial logistic regression with balanced class weights; choose C by log loss, which
  rewards calibration, not by accuracy.
- Leave the 96 trinucleotide channels out by default: on this data they lowered accuracy (0.684 vs
  0.741) and calibration (ECE 0.13 vs 0.03).
- Abstain below a confidence threshold: the lowest value (in steps of 0.05) at which out-of-fold
  accuracy on answered tumours reaches 90 %. On the benchmark this is 0.80; the model then answers
  49 % of tumours, 91 % correctly. The threshold travels inside the model file.
- Every answer carries the full probability vector, the threshold and the largest feature
  contributions, so an "uncertain" result still shows what the model leaned toward.

## Consequences
- Half of the benchmark tumours get "uncertain". That is the honest output for this data, and it is
  shown on the console and in the model card rather than tuned away.
- Explanations are exact for a linear model (coefficient × value), but they describe the model, not
  causal biology; the console says so.
- If a better-calibrated non-linear model is needed later, it must be stored safely (ADR 0001) and
  compared on the same folds with the same metrics.
