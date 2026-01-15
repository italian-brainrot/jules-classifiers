from abc import ABC, abstractmethod

import numpy as np
import polars as pl
import sklearn.semi_supervised
import torch
import torch.nn.functional as F

import utils


class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, data: pl.DataFrame, label: str, unlabeled_data: pl.DataFrame | None = None):
        # data[label] is guaranteed to be numeric, so utils.ToNumpy won't modify it.
        return self

    @abstractmethod
    def predict(self, data: pl.DataFrame) -> np.ndarray:
        ...

    def predict_proba(self, data: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError()


# --------------------------------- examples --------------------------------- #
class NearestCentroid(BaseClassifier):
    """Nearest centroid example"""
    def fit(self, data, label, unlabeled_data=None):
        self.num_cols_ = [col for col, dtype in data.schema.items() if dtype.is_numeric() and col != label]
        self.cat_cols_ = [col for col, dtype in data.schema.items() if not (dtype.is_numeric()) and col != label]

        # Save stats to normalize numeric features
        self.means_ = self.stds_ = None
        if len(self.num_cols_) > 0:
            stats = data.select([
                *[pl.col(c).mean().alias(f"{c}_mean") for c in self.num_cols_],
                *[pl.col(c).std().alias(f"{c}_std") for c in self.num_cols_]
            ]).to_dicts()[0]

            self.means_ = {c: stats[f"{c}_mean"] for c in self.num_cols_}
            self.stds_ = {c: stats[f"{c}_std"] for c in self.num_cols_}

            # Normalize
            data = data.with_columns([
                (pl.col(c) - self.means_[c]) / self.stds_[c] for c in self.num_cols_
            ])

        # Compute per-class centroids
        aggs = []
        if len(self.num_cols_) > 0: aggs.extend(pl.col(c).mean() for c in self.num_cols_)
        if len(self.cat_cols_) > 0: aggs.extend(pl.col(c).mode().first() for c in self.cat_cols_)

        self.centroids_= data.group_by(label)

        self.centroids_ = (
            data.group_by(label)
            .agg(aggs)
            .sort(label)
        )
        self.classes_ = self.centroids_[label].to_list()
        return self

    def predict_proba(self, data):
        # Normalize
        if len(self.num_cols_) > 0:
            assert self.means_ is not None and self.stds_ is not None
            data = data.with_columns([
                (pl.col(c) - self.means_[c]) / self.stds_[c] for c in self.num_cols_
            ])

        # Compute distances to centroids
        dist_cols = []
        for row in self.centroids_.to_dicts(): # type:ignore
            class_label = row.pop(list(row.keys())[0]) # The target label

            # Euclidian distance to numeric features
            sq_diff_expr = pl.sum_horizontal([
                (pl.col(c) - row[c]).pow(2) for c in self.num_cols_
            ]).sqrt() if self.num_cols_ else pl.lit(0)

            # Count of matching categorical features
            mismatch_expr = pl.sum_horizontal([
                (pl.col(c) != row[c]).cast(pl.Int32) for c in self.cat_cols_
            ]) if self.cat_cols_ else pl.lit(0)

            total_dist = (sq_diff_expr + mismatch_expr).alias(f"dist_{class_label}")
            dist_cols.append(total_dist)

        distances = data.select(dist_cols)

        # Softmax of negative distances: exp(-d_i) / sum(exp(-d_j))
        # This converts distance to a probability distribution
        prob_exprs = [
            (pl.col(c).neg().exp() / pl.sum_horizontal(pl.all().neg().exp())).alias(c.replace("dist_", ""))
            for c in distances.columns
        ]
        return distances.select(prob_exprs).to_numpy()

    def predict(self, data: pl.DataFrame):
        probs = self.predict_proba(data)
        return np.array(self.classes_)[np.argmax(probs, axis=1)]

class LabelPropagation(BaseClassifier):
    """Semi-supervised example with a sklearn estimator and unlabeled data"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, data, label, unlabeled_data=None):
        assert unlabeled_data is not None

        # Convert to numpy with scaling
        self.encoder_ = utils.ToNumpy(scale=True).fit(data, label, unlabeled_data)
        X, y = self.encoder_.transform(data)

        # LabelPropagation needs target class values with unlabeled points marked as -1
        X_unlabeled = self.encoder_.transform_X(unlabeled_data)
        X_full = np.concatenate([X, X_unlabeled], 0)
        y_full = np.concatenate([y, np.full(X_unlabeled.shape[0], -1)], 0)

        self.label_propagation_ = sklearn.semi_supervised.LabelPropagation(**self.kwargs).fit(X_full, y_full)
        return self

    def predict(self, data):
        X = self.encoder_.transform_X(data)
        return self.label_propagation_.predict(X)

    def predict_proba(self, data):
        X = self.encoder_.transform_X(data)
        p = self.label_propagation_.predict_proba(X)
        return p

class LearnableELM(BaseClassifier):
    """PyTorch example - Extreme learning machine with learnable first layer via backprop through least squares solver."""
    def __init__(self, adam_steps=500, maxiter=500, nonlinearity=F.tanh):
        self.adam_steps = adam_steps
        self.maxiter = maxiter
        self.nonlinearity = nonlinearity

    def fit(self, data, label, unlabeled_data=None):
        self.encoder_ = utils.ToNumpy(scale=True, impute=True).fit(data, label, unlabeled_data)
        X, y = self.encoder_.transform(data)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        X = torch.as_tensor(X.copy(), device=self.device, dtype=torch.float32)
        y = torch.as_tensor(y.copy(), device=self.device, dtype=torch.long)
        y_onehot = F.one_hot(y).float() # (b, n_classes) # pylint:disable=not-callable
        n_samples, n_features = X.shape
        n_clases = int(y.max() + 1)
        hidden = min(max(n_features, n_clases), 128) # hidden layer should be small, otherwise ELM overfits

        # Initialize weights
        self.W = torch.empty(hidden, n_features, device=self.device, dtype=torch.float32, requires_grad=True)
        torch.nn.init.orthogonal_(self.W)
        self.b = torch.zeros(hidden, device=self.device, dtype=torch.float32, requires_grad=True)
        self.W2 = None

        lbfgs = torch.optim.LBFGS([self.W, self.b], line_search_fn='strong_wolfe', max_iter=self.maxiter)
        adam = torch.optim.Adam([self.W, self.b], 1e-3)

        def objective():
            latents = self.nonlinearity(X @ self.W.T + self.b) # (b, hidden)

            # find weight which maps latents to targets
            self.W2 = torch.linalg.lstsq(latents, y_onehot).solution # pylint:disable=not-callable
            y_hat = latents @ self.W2
            loss = F.cross_entropy(y_hat, y) # cross_entropy expects logits
            self.W.grad = self.b.grad = None # zero grad
            loss.backward()
            return loss

        for _ in range(self.adam_steps):
            adam.step(objective)

        # LBFGS performs up to maxiter iterations per step, so step once
        lbfgs.step(objective)

        return self

    def predict(self, data):
        return np.argmax(self.predict_proba(data), axis=1)

    def predict_proba(self, data):
        X = self.encoder_.transform_X(data)
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        latents = self.nonlinearity(X @ self.W.T + self.b)
        y_hat = F.softmax(latents @ self.W2, 1)
        return y_hat.numpy(force=True)
