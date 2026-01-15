import os
import sys

from classifier import BaseClassifier
import utils

import numpy as np
import polars as pl
from sklearn.cluster import KMeans

# Implement your classifier class here,
class ProtoDashClassifier(BaseClassifier):
    def __init__(self, k=3):
        self.k = k

    def fit(self, data: pl.DataFrame, label: str, unlabeled_data: pl.DataFrame | None = None):
        self.encoder_ = utils.ToNumpy(scale=True, impute=True).fit(data, label, unlabeled_data)
        X, y = self.encoder_.transform(data)

        self.classes_ = np.unique(y)
        self.prototypes_ = [[] for _ in self.classes_]
        self.dishing_vectors_ = [[] for _ in self.classes_]

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]

            if self.k > 1 and len(X_c) >= self.k:
                kmeans = KMeans(n_clusters=self.k, random_state=0, n_init=10)
                kmeans.fit(X_c)
                prototypes = kmeans.cluster_centers_

                # Calculate dishing vectors
                if self.k > 1:
                    mean_prototype = np.mean(prototypes, axis=0)
                    dishing_vectors = mean_prototype - prototypes
                else:
                    dishing_vectors = np.zeros_like(prototypes)
            else:
                prototypes = np.mean(X_c, axis=0, keepdims=True)
                dishing_vectors = np.zeros_like(prototypes)


            self.prototypes_[i] = prototypes
            self.dishing_vectors_[i] = dishing_vectors

        return self

    def predict(self, data: pl.DataFrame) -> np.ndarray:
        probs = self.predict_proba(data)
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_proba(self, data: pl.DataFrame) -> np.ndarray:
        X = self.encoder_.transform_X(data)

        all_distances = []
        for i, _ in enumerate(self.classes_):
            class_prototypes = self.prototypes_[i]
            class_dishing_vectors = self.dishing_vectors_[i]

            distances_to_class_prototypes = []
            for prototype, dishing_vector in zip(class_prototypes, class_dishing_vectors):
                # Apply the dishing vector to the prototype
                dished_prototype = prototype - dishing_vector

                # Calculate Euclidean distance
                distances = np.linalg.norm(X - dished_prototype, axis=1)
                distances_to_class_prototypes.append(distances)

            # Find the minimum distance to any of the prototypes of this class
            min_distances = np.min(distances_to_class_prototypes, axis=0)
            all_distances.append(min_distances)

        distances_array = np.array(all_distances).T

        # Convert distances to probabilities using softmax on the negative distances
        neg_distances = -distances_array
        exp_neg_distances = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
        probs = exp_neg_distances / np.sum(exp_neg_distances, axis=1, keepdims=True)

        return probs

# Set this function to return your classifier
def get_classifier():
    return ProtoDashClassifier()

# Run
if __name__ == "__main__":
    with open(__file__, "r", encoding='utf-8') as f:
        source = f.read()

    if len(sys.argv) != 2:
        raise RuntimeError(
            f'Please specify unique name for your classifier, for example  `python {os.path.basename(__file__)} "Logistic regression"`')

    utils.run_and_save(classifier_fn=get_classifier, name=sys.argv[1], source=source)
