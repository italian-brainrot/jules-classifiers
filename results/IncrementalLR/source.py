import sys, os

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

import classifier
import utils


class IncrementalLR(classifier.BaseClassifier):
    """wrapper for estimators with sklearn-like APIs"""
    def __init__(self, estimator_fn, maxiter=10, scale=True, impute=True):
        self.estimator_fn = estimator_fn
        self.maxiter = maxiter
        self.scale, self.impute = scale, impute

    def fit(self, data, label, unlabeled_data=None):
        self.encoder_ = utils.ToNumpy(scale=self.scale, impute=self.impute).fit(data, label, unlabeled_data)
        X, y = self.encoder_.transform(data)

        self.estimators_ = []
        prev_preds = None
        for i in range(self.maxiter):
            estimator = self.estimator_fn().fit(X, y)
            preds = estimator.predict(X)
            if prev_preds is not None and np.allclose(preds, prev_preds):
                break
            prev_preds = preds

            self.estimators_.append(estimator)
            one_hot = np.zeros([X.shape[0], preds.max()+1])
            one_hot[np.arange(X.shape[0]), preds] = 1
            X = np.concatenate([X, one_hot], 1)

        return self

    def predict(self, data):
        X = self.encoder_.transform_X(data)
        preds = None
        for estimator in self.estimators_:
            preds = estimator.predict(X)
            one_hot = np.zeros([X.shape[0], preds.max()+1])
            one_hot[np.arange(X.shape[0]), preds] = 1
            X = np.concatenate([X, one_hot], 1)
        assert preds is not None
        return preds

    def predict_proba(self, data):
        X = self.encoder_.transform_X(data)
        proba = None
        for i,estimator in enumerate(self.estimators_):
            if i == len(self.estimators_) - 1:
                proba = estimator.predict_proba(X)
            else:
                preds = estimator.predict(X)
                one_hot = np.zeros([X.shape[0], preds.max()+1])
                one_hot[np.arange(X.shape[0]), preds] = 1
                X = np.concatenate([X, one_hot], 1)
        assert proba is not None
        return proba

# Run
if __name__ == "__main__":
    with open(__file__, "r", encoding='utf-8') as f:
        source = f.read()

    if len(sys.argv) != 2:
        raise RuntimeError(
            f'Please specify unique name for your classifier, for example  `python {os.path.basename(__file__)} "logistic regression"`')

    # Set classifier_fn to return your classifier
    utils.run_and_save(classifier_fn=lambda: IncrementalLR(lambda: LogisticRegression(max_iter=1000)), name=sys.argv[1], source=source)
