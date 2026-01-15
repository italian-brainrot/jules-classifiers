import os
import sys

import numpy as np
import polars as pl

import classifier
import utils


# Implement your classifier class here,
# inherit from classifier.Classifier, see classifier.py for some examples
# You can work with polars dataframe directly, or use utils.ToNumpy to convert it to numpy arrays.
# predict_proba method is optional.
class MyClassifier(classifier.Classifier):
    ...


# Run
if __name__ == "__main__":
    with open(__file__, "r", encoding='utf-8') as f:
        source = f.read()

    if len(sys.argv) != 2:
        raise RuntimeError(
            f'Please specify unique name for your classifier, for example  `python {os.path.basename(__file__)} "Logistic regression"`')

    # Set classifier_fn to return your classifier
    utils.run_and_save(classifier_fn=classifier.NearestCentroid, name=sys.argv[1], source=source)