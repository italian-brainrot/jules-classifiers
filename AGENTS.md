## Goal
The goal is to develop a novel classification algorithm which should outperform existing algorithms.

## Implementing estimators
Implement your estimator in `run.py`. It should have `fit`, `predict`, and, if applicable, `predict_proba` methods.

```python
class MyEstimator:
    def fit(self, data: pl.DataFrame, label: str, unlabeled_data: pl.DataFrame | None = None):
        # data[label] is the target, it has integer values from 0 to n_classes-1
        ...
        return self

    def predict(self, data: pl.DataFrame) -> np.ndarray:
        # Should return predictions as numpy array of shape (n_samples, )
        # with integer values from 0 to n_classes-1
        ...

    def predict_proba(self, data: pl.DataFrame) -> np.ndarray:
        # Should return probabilities as numpy array of shape (n_samples, n_targets)
        # where each row sums to 1.
        ...
```

Data is provided as polars dataframe, so you can choose how to preprocess it. If you don't want to deal with polars, you can use `utils.ToNumpy` encoder. Here is what it does:
- One-hot categorical features with more than two unique values;
- Binarizes categorical features with two unique values;
- Drops constant features (with one unique value);
- Applies StandardScaler (False by default);
- Applies SimpleImputer (False by default).

```py
encoder = utils.ToNumpy()
X, y = encoder.fit(data, label, unlabeled_data)
# X is (n_samples, n_features) array;
# y is (n_samples, ) array of integers from 0 to n_classes-1

# now new data (e.g. test data) can be transformed:
X = encoder.transform_X(data) # transform dataframe to inputs
y = encoder.transform_y(data) # transform dataframe to targets
X, y = encoder.transform(data) # transform to both
```
To enable scaling and imputation, use `utils.ToNumpy(scale=True, impute=True)`

See `classifier.py` for some examples.

## Workflow

1. Implement your estimator in `run.py`.
2. Run `python run.py "estimator name"`, where `"estimator name"` is unique name of your estimator for the leaderboard.
3. The script ranks estimators on 8 datasets via balanced accuracy, uses average rank as the final score for the leaderboard, and prints the results. It also saves results of the run to `results/*estimator_name*` and copies source code to `results/*estimator_name*/source.py`.

After each run, `LEADERBOARD.md` is automatically updated with current top-100 estimators.

## Other files

- `run_template.py` used as initial contents for `run.py`, do not modify this file.
- `utils/to_numpy.py`, `utils/workflow.py` - containts `ToNumpy` definition and utilities that handle scoring, saving and generating leaderboard. You generally shouldn't open them to save on context memory.
