"""One place that turns (task, model name) into everything the runners need. Feature set C for the tree models
(vitals snapshot + trends + core labs, plus whole-stay vitals for readmission), vitals + labs for the LSTM."""
import numpy as np

from ml.models.feature_sets import columns_for, design_matrix, load_task
from ml.models.lstm import load_sequence_data, lstm_fit_predict
from ml.models.trees import tree_fit_predict

MODEL_FEATURE_SET = {"xgboost": "C", "random_forest": "C", "lstm": "all channels + context"}


def build(task: str, model: str) -> dict:
    df, y, groups = load_task(task)
    if model == "lstm":
        data = load_sequence_data(task, "all")
        return dict(fit_predict=lstm_fit_predict(data), y=y, groups=groups, df=df, raw_columns=None, X=None)
    raw = columns_for(task, "C", df.columns)
    X = design_matrix(df, raw)
    return dict(fit_predict=tree_fit_predict(model, X.to_numpy(), y), y=y, groups=groups, df=df,
                raw_columns=raw, X=X, columns=list(X.columns))
