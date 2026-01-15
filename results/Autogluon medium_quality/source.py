import sys, os

import numpy as np
import polars as pl
from autogluon.tabular import TabularPredictor
import classifier
import utils


class AutogluonTabular(classifier.Classifier):
    def fit(self, data, label, unlabeled_data=None):
        self.predictor_ = TabularPredictor(label, verbosity=0).fit(data.to_pandas(), presets="medium_quality")
        return self

    def predict(self, data):
        return self.predictor_.predict(data.to_pandas(), as_pandas=False)

    def predict_proba(self, data):
        return self.predictor_.predict_proba(data.to_pandas(), as_pandas=False)

if __name__ == "__main__":
    with open(__file__, "r", encoding='utf-8') as f:
        source = f.read()

    if len(sys.argv) != 2:
        raise RuntimeError(
            f'Please specify unique name for your classifier, for example  `python {os.path.basename(__file__)} "logistic regression"`')

    # Set classifier_fn to return your classifier
    utils.run_and_save(classifier_fn=lambda: AutogluonTabular(), name=sys.argv[1], source=source)
