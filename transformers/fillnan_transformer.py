import time
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class FillNaNTransformer(BaseEstimator, TransformerMixin):
    """This transformer fills feature's null values."""
    def __init__(self, fill_nan_columns, train_dataset, train_target):
        self.fill_nan_columns = fill_nan_columns
        self.train_dataset = train_dataset
        self.train_target = train_target

    def transform(self, X, *_):
        if isinstance(X, pd.DataFrame):
            for column in self.fill_nan_columns:
                X[column] = X[column].fillna(self.deciles(self.train_dataset, column, self.train_target))
            print("FillNaNTransformer executed")
            return X
        else:
            raise TypeError("This transformer only works with Pandas data frames.")

    def fit(self, X, *_):
        if isinstance(X, DataFrame):
            return self
        else:
            raise TypeError("This transformer only works with Pandas data frames.")

    def deciles(self, train_dataset, column, train_target):
        """This function computes the probability of an item being "used" among those instances
        with null values in a given column. It also computes the probability of an item being
        "used" for the upper and the lower decile of that same column. If the probability of the
        null instances is closer to the probability of the upper decile (than it is to the lowest),
        it returns a value higher than that column's maximum value; otherwise, it returns a value
        lower than the minimim.
        Args: 
            train_dataset (data frame): training data set
            column (str): variable which nulls are being filled
            train_target (list): target values for items in training set
        Returns:
             Number (float) either higher than the maximum or lower than the minimum.
        """
        train_dataset['TARGET'] = [1 if y == 'used' else 0 for y in train_target]
        mean_nulos = train_dataset[train_dataset[column].isnull()]['TARGET'].mean()
        decil_sup = train_dataset[column].quantile(.90)
        maximo = train_dataset[column].max()
        decil_inf = train_dataset[column].quantile(.10)
        minimo = train_dataset[column].min()
        prob_sup = train_dataset[train_dataset[column]>decil_sup]['TARGET'].mean()
        prob_inf = train_dataset[train_dataset[column]<decil_inf]['TARGET'].mean()

        if abs(mean_nulos-prob_sup) > abs(mean_nulos-prob_inf) and minimo > 0:
            return minimo * -9999
        elif abs(mean_nulos-prob_sup) > abs(mean_nulos-prob_inf) and minimo < 0:
            return minimo * 9999
        elif abs(mean_nulos-prob_sup) > abs(mean_nulos-prob_inf) and minimo == 0:
            return -9999
        elif abs(mean_nulos-prob_sup) <= abs(mean_nulos-prob_inf) and maximo > 0:
            return maximo * 9999
        elif abs(mean_nulos-prob_sup) <= abs(mean_nulos-prob_inf) and maximo < 0:
            return maximo * -9999
        else:
            return 9999
