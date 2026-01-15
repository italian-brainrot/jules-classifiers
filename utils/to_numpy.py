import warnings
import numpy as np
import polars as pl

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

def binarize(col: pl.Series):
    unique = col.unique().to_numpy()
    mapping = {v: i for i, v in enumerate(unique.flatten())}
    return col.replace_strict(mapping, return_dtype=pl.Int8)

class ToNumpy:
    """Converts dataset to numpy arrays.

    - One-hot categorical features with more than two unique values.
    - Binarizes categorical features with two unique values.
    - Drops constant features (with one unique value).
    - If target is categorical, it is encoded to integer data type.

    Optionally this also scales and imputes features.
    Args:
        scale: whether to apply StandardScaler to features. Defaults to False.
        impute: whether to apply SimpleImputer to features. Defaults to False.
        scale_y: whether to apply MinMaxScaler to targets. Defaults to False.
        max_categories: maximum categories argument for OneHotEncoder. Defaults to 128.
    """
    def __init__(self, scale: bool=False, impute:bool=False, scale_y:bool=False, max_categories: int = 128):
        self.scale = scale
        self.impute = impute
        self.scale_y = scale_y
        self.max_categories = max_categories

    def fit(self, df: pl.DataFrame, label: str, unlabeled_data: pl.DataFrame | None = None):
        """
        Fit this transform.

        Args:
            df: dataset.
            label: name of label column.
            unlabeled_data: extra data with no label available.
        """
        self.label_ = label
        self.columns_ = df.columns
        self.columns_.remove(label)

        X = X_full = df.drop(label)
        y = df[label]

        if unlabeled_data is not None:
            X_full = pl.concat([X, unlabeled_data])

        # Preprocess X
        encoders = []
        self.binarize_cols_ = []
        self.drop_cols_ = []

        for col_name in X_full.columns.copy():
            col = X_full[col_name]

            n_unique = col.n_unique()
            if n_unique <= 1:
                # Drop constant column
                self.drop_cols_.append(col_name)
                continue

            if not col.dtype.is_numeric():
                # Binarize or one-hot encode
                if n_unique == 2: self.binarize_cols_.append(col_name)
                else: encoders.append((OneHotEncoder(sparse_output=False, max_categories=self.max_categories), [col_name]))

        self.one_hot_ = None
        if len(encoders) > 0:
            # Fit one-hot encoder if there are categorical columns
            self.one_hot_ = make_column_transformer(
                *encoders, remainder='passthrough').fit(X_full) # pyright:ignore[reportArgumentType]

        self.label_encoder_ = None
        if not (y.dtype.is_numeric() or y.dtype.is_nested()):
            # Fit label encoder if target is categorical
            self.label_encoder_ = LabelEncoder().fit(y)

        # Scale and impute
        self.scaler_ = self.imputer_ = self.y_scaler_ = None

        if self.scale or self.impute:
            # at this stage self.scaler_ and self.imputer_ are None, and won't get applied by self.transform_X
            X_full_tfm = self.transform_X(X_full)
            if self.scale: self.scaler_ = StandardScaler().fit(X_full_tfm)
            if self.impute: self.imputer_ = SimpleImputer().fit(X_full_tfm)

        if self.scale_y:
            if self.label_encoder_ is not None:
                warnings.warn("Scaling targets that are categorical, this is likely not intended "
                              "if that is the case, set scale_y to False", stacklevel=2)
            y_tfm = self.transform_y(y)
            if y_tfm.ndim == 1: y_tfm = np.expand_dims(y_tfm, 1) # MinMaxScaler needs (n_samples, n_features)
            self.y_scaler_ = MinMaxScaler().fit(y_tfm)

        return self

    def transform_X(self, df: pl.DataFrame) -> np.ndarray:
        X = df.select(self.columns_)

        # Drop constant columns
        if len(self.drop_cols_) > 0: X = X.drop(self.drop_cols_)

        # Binarize binary features
        for col_name in self.binarize_cols_: X = X.with_columns(binarize(X[col_name]))

        # One-hot encode categorical features
        if self.one_hot_ is not None: X = self.one_hot_.transform(X) # pyright:ignore[reportArgumentType]

        # Convert to numpy
        if isinstance(X, pl.DataFrame): X = X.to_numpy()
        else: X = np.asarray(X)

        # Scale and impute
        if self.scaler_ is not None: X = self.scaler_.transform(X)
        if self.imputer_ is not None: X = self.imputer_.transform(X)

        return np.asarray(X, dtype=np.float64).copy()

    def transform_y(self, df: pl.DataFrame | pl.Series) -> np.ndarray:
        if isinstance(df, pl.DataFrame): y = df[self.label_]
        else: y = df

        # Encode target if it is categorical
        if self.label_encoder_ is not None: y = self.label_encoder_.transform(y)

        # Convert to numpy
        if isinstance(y, pl.Series): y = y.to_numpy()
        else: y = np.asarray(y)

        # Scale
        if self.y_scaler_ is not None: y = self.y_scaler_.transform(y)

        return np.asarray(y).copy()

    def transform(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Apply this transform to a dataset.

        Args:
            df: dataset.

        Returns:
            X: numpy array of shape (n_features, n_targets).
            y: numpy array of shape (n_targets, ).
        """
        X = self.transform_X(df)
        y = self.transform_y(df)
        return X, y

    def fit_transform(self, df: pl.DataFrame, label: str, unlabeled_data: pl.DataFrame | None = None):
        """Fit this transform and apply it to a dataset.

        Args:
            df: dataset.

        Returns:
            X: numpy array of shape (n_features, n_targets). Doesn't include unlabeled data.
            y: numpy array of shape (n_targets, ).
        """
        self.fit(df, label, unlabeled_data)
        return self.transform(df)
