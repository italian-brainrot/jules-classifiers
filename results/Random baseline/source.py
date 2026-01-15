import sys, os

import numpy as np
import polars as pl
from scipy.special import softmax
import classifier
import utils


class RandomClassifier(classifier.Classifier):
    def fit(self, data, label, unlabeled_data=None):
        self.rng_ = np.random.default_rng(0)
        self.n_classes_ = data[label].n_unique()
        return self

    def predict(self, data):
        return self.rng_.integers(0,self.n_classes_, size=len(data))

    def predict_proba(self, data):
        logits = self.rng_.uniform(-1, 1, (len(data), self.n_classes_))
        return softmax(logits, 1)

if __name__ == "__main__":
    with open(__file__, "r", encoding='utf-8') as f:
        source = f.read()

    if len(sys.argv) != 2:
        raise RuntimeError(
            f'Please specify unique name for your classifier, for example  `python {os.path.basename(__file__)} "logistic regression"`')

    # Set classifier_fn to return your classifier
    utils.run_and_save(classifier_fn=lambda: RandomClassifier(), name=sys.argv[1], source=source)
