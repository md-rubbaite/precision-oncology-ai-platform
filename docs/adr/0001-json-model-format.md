# ADR 0001: Models are JSON, scored with NumPy; no pickle

- Status: accepted
- Date: 2026-11-24

## Context
The usual way to ship a scikit-learn model is `pickle` or `joblib`. Loading a pickle executes code
chosen by whoever wrote the file, so a model file becomes an attack vector for anyone who can
replace it (a shared drive, an artifact store, a pull request). Research groups also need to see
what a model contains: which genes it uses, how it was evaluated, on which data.

## Decision
- The served model is multinomial logistic regression. Its state is small: class names, the feature
  space (gene list, protein-change list, scaling), a coefficient matrix and intercepts.
- It is written as JSON (`format: oncoai-model/1`) together with its evaluation summary and data
  source, and scored with NumPy. scikit-learn is used only to fit it.
- Loading checks the format tag, file size (20 MB), shapes, finite values and the threshold range.
- Binary models are expanded to the softmax form so scoring is the same for any number of classes.

## Consequences
- A model file can be reviewed in a text editor and diffed in Git. Loading cannot run code.
- Serving does not need scikit-learn; the container is smaller and has fewer dependencies to patch.
- Model families that cannot be written this way (gradient boosting, neural networks) would need a
  safe format such as ONNX or skops with an allow-list. ADR 0003 explains why the linear model is
  the one served today.
- A test checks that NumPy scoring equals scikit-learn's `predict_proba` to 1e-8.
