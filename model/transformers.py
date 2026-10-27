
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MissingFilter(BaseEstimator, TransformerMixin):
    def __init__(self, max_missing=0.4): self.max_missing = max_missing
    def fit(self, X, y=None):
        self.keep_ = pd.DataFrame(X).isna().mean().values <= self.max_missing
        return self
    def transform(self, X): return np.asarray(X, dtype=float)[:, self.keep_]
    def get_feature_names_out(self, input_features=None): return np.asarray(input_features)[self.keep_]

class CorrFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95): self.threshold = threshold
    def fit(self, X, y=None):
        c = np.nan_to_num(np.abs(np.corrcoef(np.asarray(X, float), rowvar=False)))
        self.keep_ = ~(np.triu(c, k=1) > self.threshold).any(axis=0)
        return self
    def transform(self, X): return np.asarray(X, dtype=float)[:, self.keep_]
    def get_feature_names_out(self, input_features=None): return np.asarray(input_features)[self.keep_]
