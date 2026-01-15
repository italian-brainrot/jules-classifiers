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
class RecursiveFeatureHalvingClassifier(classifier.Classifier):
    def __init__(self, max_depth=10, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None
        self.encoder_ = None

    def fit(self, data: pl.DataFrame, label: str, unlabeled_data: pl.DataFrame | None = None):
        self.encoder_ = utils.ToNumpy(scale=True, impute=True).fit(data, label, unlabeled_data)
        X, y = self.encoder_.transform(data)
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, data: pl.DataFrame) -> np.ndarray:
        X = self.encoder_.transform_X(data)
        predictions = np.array([self._traverse_tree(x, self.tree_) for x in X])
        return predictions

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or
                n_samples < self.min_samples_leaf or
                n_classes == 1):
            leaf_value = np.argmax(np.bincount(y))
            return leaf_value

        # PCA to find the split direction
        if n_samples > 1 and n_features > 0:
            # Centering the data
            mean = np.mean(X, axis=0)
            X_centered = X - mean

            # SVD for PCA
            try:
                _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
                pc1 = vt[0, :]
            except np.linalg.LinAlgError:
                # If SVD fails, fallback to a random split
                pc1 = np.random.randn(n_features)
                pc1 /= np.linalg.norm(pc1)

            # Project data and find split point
            projections = X_centered @ pc1
            median = np.median(projections)

            # Split data
            left_mask = projections < median
            right_mask = ~left_mask

            # Check for empty splits
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                leaf_value = np.argmax(np.bincount(y))
                return leaf_value

            # Recursive calls
            left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
            right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

            return {'pc': pc1, 'median': median, 'left': left_child, 'right': right_child, 'mean': mean}
        else:
            leaf_value = np.argmax(np.bincount(y))
            return leaf_value

    def _traverse_tree(self, x, node):
        if isinstance(node, dict):
            projection = (x - node['mean']) @ node['pc']
            if projection < node['median']:
                return self._traverse_tree(x, node['left'])
            else:
                return self._traverse_tree(x, node['right'])
        else:
            return node


# Run
if __name__ == "__main__":
    with open(__file__, "r", encoding='utf-8') as f:
        source = f.read()

    if len(sys.argv) != 2:
        raise RuntimeError(
            f'Please specify unique name for your classifier, for example  `python {os.path.basename(__file__)} "Logistic regression"`')

    # Set classifier_fn to return your classifier
    utils.run_and_save(classifier_fn=lambda: RecursiveFeatureHalvingClassifier(), name=sys.argv[1], source=source)