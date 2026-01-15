import sys, os

import numpy as np
import polars as pl
from sklearn.neighbors import NearestCentroid

import classifier
import utils


class SklearnEstimator(classifier.Classifier):
    """wrapper for estimators with sklearn-like APIs"""
    def __init__(self, estimator, scale=True, impute=True):
        self.estimator = estimator
        self.scale, self.impute = scale, impute

    def fit(self, data, label, unlabeled_data=None):
        self.encoder_ = utils.ToNumpy(scale=self.scale, impute=self.impute).fit(data, label, unlabeled_data)
        X, y = self.encoder_.transform(data)
        self.estimator.fit(X, y)
        return self

    def predict(self, data):
        X = self.encoder_.transform_X(data)
        return self.estimator.predict(X)

    def predict_proba(self, data):
        X = self.encoder_.transform_X(data)
        return self.estimator.predict_proba(X)

# Run
if __name__ == "__main__":
    with open(__file__, "r", encoding='utf-8') as f:
        source = f.read()

    if len(sys.argv) != 2:
        raise RuntimeError(
            f'Please specify unique name for your classifier, for example  `python {os.path.basename(__file__)} "logistic regression"`')

    # Set classifier_fn to return your classifier
    utils.run_and_save(classifier_fn=lambda: SklearnEstimator(NearestCentroid()), name=sys.argv[1], source=source)
