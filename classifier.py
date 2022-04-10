import time
import pandas as pd

from settings import hyper_parameters, fill_nan_columns, categoric_columns, features

from transformers.fillnan_transformer import FillNaNTransformer
from transformers.dummies_transformer import DummiesTransformer
from transformers.features_selector import FeaturesSelector

from sklearn.pipeline  import Pipeline
from sklearn.ensemble import RandomForestClassifier


class ClassificationPredictor:
    def __init__(self):
        self._pipeline = None
        
    def fit(self, X, y):
        self._pipeline.fit(X, y)

    def add_pipeline_step(self, step):
        """Add a step to the pipeline.
        Args:
            step (tuple): The first element should be the name of the step and the
            second element should be the function to execute.
        Returns:
            None
        """
        if not self._pipeline:
            self._pipeline = Pipeline(steps=[step])
        else:
            self._pipeline.steps.append(step)

    def predict(self, input):
        """Obtain the model's inference from the given input."""
        return self._pipeline.predict(input)


def train(model, X_train, y_train):
    """Train the model instance, from the given dataset."""

    model.add_pipeline_step(('fill_nan_transformers', FillNaNTransformer(fill_nan_columns, X_train, y_train)))
    model.add_pipeline_step(('dummies_transformer', DummiesTransformer(categoric_columns)))
    model.add_pipeline_step(('columns_selector', FeaturesSelector(features)))
    model.add_pipeline_step(('model', RandomForestClassifier(**hyper_parameters)))
    model.fit(X_train, y_train)

    return model