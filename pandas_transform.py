from sklearn.base import BaseEstimator, TransformerMixin


class PandasTransform(TransformerMixin, BaseEstimator):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):
        return self.fn(X)