import os
import pandas as pd
import pytest
from ml.data import process_data
from ml.model import train_model, inference, performance_on_categorical_slice

@pytest.fixture(scope="module")
def data():
    return pd.read_csv(os.path.join("data", "census.csv"))

def test_process_data_output_shape(data):
    """
    Test that process_data returns correct shapes for inputs and labels.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    assert X.shape[0] == y.shape[0], "Mismatch between features and labels"
    assert X.shape[1] > 0, "No features returned"

def test_model_training(data):
    """
    Test that model trains and can produce predictions.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(y), "Predictions and labels count mismatch"

def test_slice_performance(data):
    """
    Test that slice performance returns metrics on a known column value.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    p, r, f = performance_on_categorical_slice(
        data, column_name="education", slice_value="Bachelors",
        categorical_features=cat_features, label="salary",
        encoder=encoder, lb=lb, model=model
    )
    assert 0 <= p <= 1 and 0 <= r <= 1 and 0 <= f <= 1, "Invalid metric values"
