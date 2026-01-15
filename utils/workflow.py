import json
import time
from collections import defaultdict
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mnist1d.data
import numpy as np
import openml
import polars as pl
import polars.selectors as cs
import scipy.sparse
import sklearn.datasets
import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data  # type:ignore

if TYPE_CHECKING:
    from ..classifier import Classifier

def evaluate_classifier_on_dataset(
    classifier_fn: "Callable[..., Classifier]",
    dataset: pl.DataFrame,
    label: str,
    n_folds=2,
) -> tuple[pl.DataFrame, pl.DataFrame, defaultdict[str, list[float]], defaultdict[str, list[float]]]:

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    X = dataset.drop(label)
    y = dataset.select(label)
    y = pl.Series(label, LabelEncoder().fit_transform(y.to_series())).to_frame() # make sure y is encoded to integer for metrics

    metrics_train = defaultdict(list)
    metrics_test = defaultdict(list)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)): # type:ignore
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        df_fold = pl.concat((X_train, y_train), how='horizontal')
        predictor = classifier_fn().fit(df_fold, label, unlabeled_data=X_test)

        # Compute metrics
        for data, labels, metrics in [(X_train, y_train, metrics_train), (X_test, y_test, metrics_test)]:
            preds = predictor.predict(data)
            metrics["accuracy"].append(accuracy_score(np.squeeze(labels), np.squeeze(preds)))
            metrics["balanced_accuracy"].append(balanced_accuracy_score(np.squeeze(labels), np.squeeze(preds)))
            average = 'binary' if y.max().item() == 1 else "weighted"
            metrics["precision"].append(precision_score(np.squeeze(labels), np.squeeze(preds), average=average, zero_division=0))
            metrics["recall"].append(recall_score(np.squeeze(labels), np.squeeze(preds), average=average, zero_division=0))

            # Compute ROC AUC
            try:
                probs = predictor.predict_proba(data)
                if np.isfinite(probs).all():
                    if probs.ndim == 2 and probs.shape[1] == 2: probs = probs[:, 1] # handle binary
                    metrics["roc_auc"].append(roc_auc_score(np.squeeze(labels), np.nan_to_num(probs), multi_class='ovr'))
            except (NotImplementedError, AttributeError):
                pass

    return X, y, metrics_train, metrics_test

def _tofloat(x):
    x = np.asarray(x)
    if x.size > 1: x = np.mean(x)
    if x.size == 0: return np.nan
    return x.item()

def load_mnist1d():
    defaults = mnist1d.data.get_dataset_args()
    data = mnist1d.data.make_dataset(defaults)
    # X is a (num_samples, 40);
    # y is (num_samples, )
    return pl.from_numpy(data["x"]).with_columns(pl.Series("y", data["y"]))

def load_mnist_subsampled():
    task = openml.tasks.get_task(3573)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    X = pl.from_pandas(X) # type:ignore
    y = pl.Series(y).to_frame()
    df = pl.concat([X, y], how='horizontal')
    return df.group_by(dataset.default_target_attribute).map_groups(lambda df: df.sample(100, shuffle=True, seed=0)), dataset.default_target_attribute

def load_iris_useless_features():
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X) # type:ignore
    # add ~50 random features
    rng = np.random.default_rng(0)
    feats1 = rng.standard_normal((X.shape[0], 10))
    feats2 = feats1 + rng.standard_normal((X.shape[0], 10)) * 0.01
    feats3 = scipy.sparse.random(X.shape[0], 10, rng=rng).toarray()
    feats4 = np.outer(rng.standard_normal(X.shape[0]), rng.standard_normal(10))
    feats5 = X + rng.standard_normal((X.shape[0], X.shape[1])) * 0.01
    X = np.concatenate([X, feats1, feats2, feats3, feats4, feats5], 1)
    X = pl.from_numpy(X) # type:ignore
    y = pl.Series("y", y).to_frame()
    df = pl.concat([X, y], how='horizontal')
    return df

def yield_datasets() -> Generator[tuple[str, pl.DataFrame, str]]:
    ids = (
        9946, # wdbc, easy
        37, # diabetes, medium
        3913, # kc2 small centroid advantage
        3918, # pc1 small forest advantage
        49, # tic-tac-toe, nonlinear
    )
    for task_id in ids:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = pl.from_pandas(X) # type:ignore
        y = pl.Series(y).to_frame()
        df = pl.concat([X, y], how='horizontal')
        yield (dataset.name, df, cast(str, dataset.default_target_attribute))

    yield ("mnist1d", load_mnist1d(), "y")

    df, label = load_mnist_subsampled()
    yield ("MNIST-subsampled", df, cast(str, label))

    yield ("iris+useless features", load_iris_useless_features(), "y")

def evaluate_classifier(classifier_fn: "Callable[..., Classifier]", ):
    """Evaluates classifier on some datasets from OpenML-CC18 and returns metrics as dictionary ``{dataset_name: {metric: value}}``"""

    metrics: dict[str, dict[str, float]] = {}
    summary = []
    for name, df, label in yield_datasets():
        assert label is not None

        X, y, metrics_train, metrics_test = evaluate_classifier_on_dataset(classifier_fn, df, label)
        metrics[name] = {k: np.mean(v).item() for k,v in metrics_test.items()}
        # print(f'{dataset.name} with {X.shape[0]} samples, {X.shape[1]} features, {y.max().item()+1} classes: {_format_metrics(metrics_train, metrics_test)}')
        row = {
            "dataset": name,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": y.max().item()+1,
        }
        for k,v in metrics_train.items():
            if k == "accuracy": continue
            row[f'{k} (train/test)'] = f"{_tofloat(v):.4g} / {_tofloat(metrics_test[k]):.4g}"
        summary.append(row)

    with pl.Config(
        tbl_hide_dataframe_shape=True,
        set_tbl_hide_column_data_types=True,
        set_float_precision=4,
        tbl_cols=20,
        tbl_formatting="MARKDOWN",
        tbl_hide_dtype_separator=True,
        set_fmt_str_lengths=100
    ):
        print(f"{classifier_fn().__class__.__name__} metrics:")
        print(pl.from_dicts(summary))

    return metrics


def sanity_check(classifier_fn: "Callable[..., Classifier]"):
    """Fits classifier to a very easy linearly-separable dataset prints some outputs."""
    # 8 points with 2 linearly-separable classes
    x1 = np.concatenate([np.linspace(0, 1, 100) + 1, np.linspace(0, 1, 100) - 1])
    x2 = np.concatenate([np.linspace(0, 1, 100) - 1, np.linspace(0, 1, 100) + 1])
    x1 = (x1 - x1.mean()) / x1.std()
    x2 = (x2 - x2.mean()) / x2.std()
    y = np.argmax(np.stack([x1, x2], 1), 1)

    df = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
    df_train = df[::2]
    df_test = df[1::2]

    start = time.perf_counter()
    predictor = classifier_fn()
    print(f"Sanity check: fitting {predictor.__class__.__name__} on linearly separable y=argmax(x1, x2)")
    predictor.fit(df_train, "y", unlabeled_data=df_test.drop("y"))
    print(f'Fitting {df_train.shape} took {(time.perf_counter() - start):.2g} sec.')

    # Print predictions for inspection
    df_full = pl.concat([
        df_train.with_columns(split=pl.lit("train")),
        df_test.with_columns(split=pl.lit("test"))
    ])

    preds = predictor.predict(df_full.drop("y", "split"))
    summary = df_full.with_row_index().with_columns(
        preds=pl.Series("preds", preds),
    ).drop("index").select("split","x1","x2","y","preds")

    probas = None
    try:
        probas = predictor.predict_proba(df_full.drop("y", "split"))
        summary = summary.with_columns(probabilities=pl.Series("probabilities", probas))
    except (NotImplementedError, AttributeError):
        print(f'{predictor.__class__.__name__}.predict_proba is not implemented.')

    with pl.Config(
        tbl_hide_dataframe_shape=True,
        set_tbl_hide_column_data_types=True,
        set_float_precision=3,
        tbl_formatting="MARKDOWN",
        tbl_hide_dtype_separator=True,
    ):
        print(f"Predictions for few samples:\n{summary[10,30,60,90,25,75]}")
    print(f"Sanity check accuracy: {(summary["preds"]==summary["y"]).mean()}.")

    # Some checks
    if preds.shape != (len(df_full), ):
        raise RuntimeError(f"{predictor.__class__.__name__}.predict returned array of shape {preds.shape} "
                           f"for dataframe of shape {df_full.shape}. The shape should be (n_samples, ): {(len(df_full), )}.")

    if np.isin(preds, (0,1), invert=True).any():
        raise RuntimeError(f"{predictor.__class__.__name__}.predict returned predictions with unique values {np.unique(preds)} "
                           "for a binary classification dataset. It should return integers in range [0, n_classes - 1].")

    if probas is not None:
        if probas.shape != (len(df_full), 2):
            raise RuntimeError(f"{predictor.__class__.__name__}.predict_probas returned array of shape {probas.shape}. "
                               f"for dataframe of shape {df_full.shape}. The shape should be (n_samples, n_classes): {(len(df_full), 2)}.")


class SklearnAdapter(ClassifierMixin, BaseEstimator):
    """adapter to use Classifier as sklearn estimator."""
    def __init__(self, classifier: "Classifier"):
        self.classifier = classifier

    def fit(self, X, y, X_unlabeled=None):
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)

        self.is_fitted_ = True
        if isinstance(X, np.ndarray): X = pl.from_numpy(X)
        if isinstance(X_unlabeled, np.ndarray): X_unlabeled = pl.from_numpy(X_unlabeled)
        if isinstance(y, np.ndarray): y = pl.Series(np.squeeze(y))
        if isinstance(y, pl.DataFrame): y = y.to_series()
        y = y.rename("class").to_frame()
        df = pl.concat([X,y], how='horizontal')
        self.classifier.fit(df, "class", X_unlabeled)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if isinstance(X, np.ndarray): X = pl.from_numpy(X)
        return self.classifier.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if isinstance(X, np.ndarray): X = pl.from_numpy(X)
        return self.classifier.predict_proba(X)

cm_bright = mcolors.ListedColormap(["#FF0000", "#0000FF"])

def plot_decision_boundary(
    classifier: "Classifier",
    X: np.ndarray,
    y: np.ndarray,
    X_unlabeled: np.ndarray,
    grid_resolution: int = 200,
    proba: bool | str = 'auto',
    plot_method="pcolormesh",
    cmap=cm_bright,
    s=2,
    alpha=0.5,
):
    """Fits classifier to X and y, and plots decision boundary"""
    estimator = SklearnAdapter(classifier).fit(X, y, X_unlabeled)
    if proba == 'auto':
        if y.max() > 1:
            proba=False
        else:
            try:
                p = estimator.predict_proba(X)
                proba = p.shape[1] > 1
            except (NotImplementedError, AttributeError):
                proba=False

    response_method = "predict_proba" if proba else "predict"

    kw:dict = dict(cmap=cmap)
    if y.max() > 1: kw = {} # removes warning in multiclass case

    disp = DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        eps = (X.max() - X.min()) / 2,
        grid_resolution=grid_resolution,
        response_method=response_method, # pyright:ignore[reportArgumentType]
        plot_method=plot_method, # pyright:ignore[reportArgumentType]
        **kw,
    )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=np.squeeze(y), edgecolor="k", s=s, alpha=alpha, linewidths=s/20, **kw)
    return disp

# twospirals from https://glowingpython.blogspot.com/2017/04/solving-two-spirals-problem-with-keras.html
def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    rng = np.random.default_rng(0)
    n = np.sqrt(rng.uniform(0, 1, (n_points,1))) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + rng.uniform(0, 1, (n_points,1)) * noise
    d1y = np.sin(n)*n + rng.uniform(0, 1, (n_points,1)) * noise
    return (
        np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
        np.hstack((np.zeros(n_points), np.ones(n_points))).astype(np.int64)
        )

def periodic(n_points):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 5, (n_points, 2))
    y = ((X[:,0] % 1) > 0.5).astype(np.int64) + ((X[:,1] % 1) > 0.5).astype(np.int64)
    return X, y

def save_visualizations(dir, classifier_fn: "Callable[..., Classifier]"):
    """Generates and saves decision boundaries of classifier on some datasets"""
    def save(X,y,name,s,alpha):
        X, X_unlabeled = X[::2], X[1::2]
        y = y[::2]
        disp = plot_decision_boundary(classifier_fn(), X, y, X_unlabeled, s=s, alpha=alpha)
        disp.figure_.set_layout_engine("compressed")
        disp.figure_.set_size_inches(8,8)
        disp.figure_.savefig(Path(dir) / f"{name}.png")
        plt.close()

    X,y = sklearn.datasets.make_blobs(2000, random_state=0, centers=10) # pylint:disable=unbalanced-tuple-unpacking # type:ignore
    save(X, y, "blobs", s=10, alpha=0.8)

    X,y = sklearn.datasets.make_circles(200, random_state=0)
    save(X, y, "circles", s=10, alpha=0.5)

    X,y = sklearn.datasets.make_moons(200, random_state=0)
    save(X, y, "moons", s=10, alpha=0.5)

    X,y = sklearn.datasets.make_moons(2000, noise=0.2, random_state=0)
    save(X, y, "moons-noisy", s=10, alpha=0.5)

    X,y = sklearn.datasets.make_moons(2000, noise=0.4, random_state=0)
    save(X, y, "moons-very-noisy", s=10, alpha=0.5)

    X,y = twospirals(2000, 1)
    save(X, y, "spirals", s=10, alpha=0.5)

    X,y = periodic(2000)
    save(X, y, "periodic", s=10, alpha=0.5)

def load_metrics_to_frame():
    root = Path(__file__).parent.parent
    results = root / "results"

    data = []
    for run in results.iterdir():
        with open(run/"metrics.json", "r", encoding='utf-8') as f:
            datasets: dict[str, dict[str, float]] = json.load(f)

        # datasets is ``{dataset_name: {metric: value}}``, flatten
        metrics_flat: dict[str, Any] = {f"{dataset}-{metric}": value for dataset, metrics in datasets.items() for metric, value in metrics.items()}
        for metric_name in next(iter(datasets.values())):
            metric = [v for k,v in metrics_flat.items() if k.endswith(f"-{metric_name}")]
            metrics_flat[f"mean_{metric_name}"] = float(np.mean(metric))

        metrics_flat["name"] = run.name
        data.append(metrics_flat)

    df = pl.DataFrame(data)

    # add ranks based on balanced accuracy
    for col_name in df.columns:
        if "-balanced_accuracy" in col_name:
            df = df.with_columns(pl.col(col_name).neg().rank().alias(f"{col_name.replace('-balanced_accuracy', '-rank')}"))

    df = pl.DataFrame(data).with_columns(pl.mean_horizontal(df.select(cs.contains("-rank"))).alias("avg_rank"))
    df = df.sort("avg_rank", descending=False).with_row_index("#", offset=1)
    return df

def run_and_save(classifier_fn: "Callable[..., Classifier]", name: str, source:str):
    name = name.strip()
    if len(name) == 0:
        raise RuntimeError("Name is empty, please specify unique name of the classifier")

    # Check that dir doesn't exist
    root = Path(__file__).parent.parent
    results = root / "results"

    run_dir = results / name
    if run_dir.exists():
        raise FileExistsError(f'Result with name "{name}" already exists at "{run_dir}".\n'
                              'If this is a different classifier, provide a different name, or rename existing directory.\n'
                              'If this directory represents a failed run, you can delete it and rerun with this name.')

    # Sanity check
    sanity_check(classifier_fn)

    # Run and evaluate
    print(f'\nEvalutating "{name}" on 8 datasets...')
    metrics = evaluate_classifier(classifier_fn)

    # Create run dir
    if not results.exists(): results.mkdir()
    run_dir.mkdir()

    # Save source and metrics
    with open(run_dir / "source.py", "w", encoding='utf8') as f: f.write(source)
    with open(run_dir / 'metrics.json', 'w', encoding='utf8') as f: json.dump(metrics, f, indent=4) #

    # Make visualizations
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir()
    save_visualizations(vis_dir, classifier_fn)

    # Save leaderboard
    leaders = load_metrics_to_frame()
    with pl.Config(
        tbl_hide_dataframe_shape=True,
        set_tbl_hide_column_data_types=True,
        set_float_precision=5,
        tbl_formatting="MARKDOWN",
        tbl_hide_dtype_separator=True,
        set_fmt_str_lengths=100,
        set_tbl_rows=100,
    ):
        with open(root / "LEADERBOARD.md", "w", encoding='utf-8') as f:
            print(f"{leaders.select("#", "name", "avg_rank", "mean_balanced_accuracy", "mean_roc_auc").head(100)}", file=f)

        # Print results
        row = leaders.filter(pl.col("name")==name)
        rank = row["#"].to_list()[0]
        print(f"\nFinished, reached #{rank}/{len(leaders)} in leaderboard:")
        head = leaders.head(10)
        if len(head.filter(pl.col("name")==name)) == 0:
            head = pl.concat([head, row])
        print(head.select("#", "name", "avg_rank", "mean_balanced_accuracy", "mean_roc_auc"))

        print()
        print("Best classifiers per dataset:")
        data = []
        for dataset_name in metrics:
            dataset_df = leaders.select("#", "avg_rank", "name", f"{dataset_name}-balanced_accuracy", f"{dataset_name}-roc_auc", "mean_balanced_accuracy", "mean_roc_auc")
            dataset_df_bal_acc = dataset_df.drop_nulls(f"{dataset_name}-balanced_accuracy").sort(f"{dataset_name}-balanced_accuracy", descending=True).with_row_index("dataset_position", 1)
            dataset_df_roc_auc = dataset_df.drop_nulls(f"{dataset_name}-roc_auc").sort(f"{dataset_name}-roc_auc", descending=True).with_row_index("dataset_position", 1)

            def _get0(x,s):
                return x[0][s].to_list()[0]

            bal_acc_col = dataset_df_bal_acc.filter(pl.col("name")==name)
            roc_auc_col = dataset_df_roc_auc.filter(pl.col("name")==name)
            row = {
                "dataset": dataset_name,
                "top_model_by_balanced_acc":
                    f'#{_get0(dataset_df_bal_acc, "#")}: {_get0(dataset_df_bal_acc, "name")}: '
                    f'{_get0(dataset_df_bal_acc, f"{dataset_name}-balanced_accuracy"):.5g}',
                f"{name[:20]} balanced acc":
                    f'{bal_acc_col[f"{dataset_name}-balanced_accuracy"].to_list()[0]:.5g} '
                    f'(rank {bal_acc_col["dataset_position"].to_list()[0]})',
                "top_model_by_roc_auc":
                    f'#{_get0(dataset_df_roc_auc, "#")}: {_get0(dataset_df_roc_auc, "name")}: '
                    f'{_get0(dataset_df_roc_auc, f"{dataset_name}-roc_auc"):.5g}',
            }
            if len(roc_auc_col) > 0:
                row.update({
                    f"{name[:20]} ROC AUC":
                        f'{roc_auc_col[f"{dataset_name}-roc_auc"].to_list()[0]:.5g} (rank {roc_auc_col["dataset_position"].to_list()[0]})'
                })
            data.append(row)

        print(pl.from_dicts(data))
