import time

from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

class FeaturesSelector(BaseEstimator, TransformerMixin):
    """This transformer selects model's features from data frame."""
    def __init__(self, features):
        self.features = features

    def transform(self, X, *_):
        if isinstance(X, DataFrame):
            X = X[self.features]
            print("FeaturesSelector excecuted")
            return X
        else:
            raise TypeError("This transformer only works with Pandas data frames.")

    def fit(self, X, *_):
        if isinstance(X, DataFrame):
            return self
        else:
            raise TypeError("This transformer only works with Pandas data frames.")