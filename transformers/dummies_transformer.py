import time

from pandas import DataFrame, get_dummies
from sklearn.base import BaseEstimator, TransformerMixin


class DummiesTransformer(BaseEstimator, TransformerMixin):
    """This transformers responsibility is to transform a column like this:
    
    TAG
    ===
    A
    C
    B
    A
    C
    
    Into a set of columns like this
    
    TAG_A | TAG_B | TAG_C
    =====================
      1   |   0   | 0
      0   |   0   | 1
      0   |   1   | 0
      1   |   0   | 0
      0   |   0   | 1
    """
    def __init__(self, categoric_columns):
        self.categoric_columns = categoric_columns
        self.dummies_columns = None

    def transform(self, X, *_):
        if self.categoric_columns is None or len(self.categoric_columns) == 0:
            return X
        if isinstance(X, DataFrame):
            X = get_dummies(X, columns=self.categoric_columns, dummy_na = True)
            for missing_column in [x for x in list(self.dummies_columns) if x not in list(X.columns)]:
                X[missing_column] = 0
            print("DummiesTransformer excecuted")
            return X
        else:
            raise TypeError("This transformer only works with Pandas data frames.")

    def fit(self, X, *_):
        if self.categoric_columns is None or len(self.categoric_columns) == 0:
            return self
        if isinstance(X, DataFrame):
            X_dummies = get_dummies(X, columns=self.categoric_columns, dummy_na = False)
            self.dummies_columns = X_dummies.columns
            return self
        else:
            raise TypeError("This transformer only works with Pandas data frames.")