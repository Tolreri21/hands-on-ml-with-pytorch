# Hands-On Machine Learning — Study Notes

My working notes and code from *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*
by Aurélien Géron. One folder per chapter, written while reading: I retype the examples instead of
copying them, break things on purpose, and add my own experiments on top.

This is a learning journal, not a library — notebooks are meant to be read top to bottom.

## Stack

- Python 3.12, dependencies managed with [uv](https://docs.astral.sh/uv/)
- NumPy, pandas, scikit-learn
- matplotlib, seaborn
- Jupyter notebooks (via `ipykernel`, run from PyCharm)

## Setup

```bash
git clone <repo-url>
cd hands-on-ml-with-pytorch
uv sync
```

Notebooks are opened in PyCharm using the project's `.venv` kernel. For a browser UI instead:

```bash
uv add --dev jupyterlab
uv run jupyter lab
```

Plain scripts import across folders, so run them from the project root with it on the path:

```bash
PYTHONPATH=. uv run python chapter1/example1-1.py
```

## Structure

```
chapter1/   from-scratch linear regression
chapter2/   end-to-end project: California housing + a salary prediction side project
chapter3/   classification on MNIST
```

## Chapters

### Chapter 1 — The Machine Learning Landscape

- `LinearRegression.py` — linear regression implemented from scratch on NumPy: batch gradient
  descent, weights/bias updates by hand, MSE.
- `example1-1.py` — the book's life-satisfaction vs. GDP example, fitted with my own class instead
  of scikit-learn. Lesson learned the hard way: without feature normalization gradient descent blows up.

### Chapter 2 — End-to-End Machine Learning Project

- `chapter2_code.ipynb` — the full California housing walkthrough: EDA and histograms, a hand-written
  train/test split plus hash-based splitting, stratified sampling on income categories, correlation
  analysis, engineered ratio features (rooms per household, etc.), `SimpleImputer` / `OneHotEncoder` /
  scalers, a custom RBF-kernel cluster-similarity transformer, `ColumnTransformer` + `Pipeline`.
  Models: linear regression → decision tree → random forest, compared with cross-validation, then
  `GridSearchCV` and `RandomizedSearchCV`, feature importances, final evaluation on the test set with
  a bootstrap confidence interval for the RMSE.
- `salary_prediction_project.ipynb` — my own take on the Adult census dataset (predicting income
  `>50K`): EDA with seaborn, `?` treated as missing data, per-category target rates, then a
  preprocessing pipeline over numeric and categorical columns.
- `datasets/` — housing data downloaded by the notebook, plus `adult.csv`.

### Chapter 3 — Classification

- `classification.ipynb` — MNIST fetched from OpenML. Binary "is it a 5?" classifier with
  `SGDClassifier`, why accuracy lies on skewed data (compared against a `DummyClassifier`), confusion
  matrix, precision / recall / F1, moving the decision threshold by hand, precision–recall curve,
  picking a threshold for a target precision, ROC curve and ROC AUC, and finally a
  `RandomForestClassifier` compared against SGD on both curves.

## Notes to self

- Accuracy is a bad metric when one class dominates — check the dummy baseline first.
- Precision/recall is a dial, not a fixed property: pick the threshold from the curve, based on what
  the task actually costs.
- Always fit preprocessing inside the pipeline, so cross-validation doesn't leak the test fold.
