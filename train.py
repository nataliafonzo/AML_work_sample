import time
import pandas as pd

from settings import hyper_parameters, fill_nan_columns, categoric_columns, features

from transformers.fillnan_transformer import FillNaNTransformer
from transformers.dummies_transformer import DummiesTransformer
from transformers.features_selector import FeaturesSelector

from model import ClassificationPredictor
from sklearn.ensemble import RandomForestClassifier


def train(model, X_train, y_train):
    """Train the model instance, from the given dataset."""

    model.add_pipeline_step(('fill_nan_transformers', FillNaNTransformer(fill_nan_columns, X_train, y_train)))
    model.add_pipeline_step(('dummies_transformer', DummiesTransformer(categoric_columns)))
    model.add_pipeline_step(('columns_selector', FeaturesSelector(features)))
    model.add_pipeline_step(('model', RandomForestClassifier(**hyper_parameters)))
    model.fit(X_train, y_train)

    return model