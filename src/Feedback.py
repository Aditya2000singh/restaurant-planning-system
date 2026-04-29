import joblib
import numpy as np
import pandas as pd
import os
from predictor import get_features, MODEL_PATH, SCALER_PATH

CORRECTIONS_LOG = "data/corrections.csv"


def apply_correction(date, day_of_week, is_raining, is_holiday, is_event,
                     predicted_covers, actual_covers, reason=""):
    """
    This is the main feedback function.
    When a manager says "we predicted 120 but got 85 because of rain",
    we call this — it updates the model on that single data point.

    SGDRegressor.partial_fit() is what makes it self-learning.
    It doesn't retrain from scratch, just nudges the weights.
    """
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X = get_features(day_of_week, is_raining, is_holiday, is_event)
    X_scaled = scaler.transform(X)

    # update model with the real value
    model.partial_fit(X_scaled, [actual_covers])

    # save updated model back
    joblib.dump(model, MODEL_PATH)
    print(f"Model updated: predicted={predicted_covers}, actual={actual_covers}")

    # log correction so we can track how the model is learning
    _log_correction(date, predicted_covers, actual_covers, reason)


def _log_correction(date, predicted, actual, reason):
    """Keeps a CSV log of every correction managers make."""
    error = abs(predicted - actual)
    pct_error = round((error / actual) * 100, 1) if actual > 0 else 0

    row = {
        "date":      date,
        "predicted": predicted,
        "actual":    actual,
        "error":     error,
        "pct_error": pct_error,
        "reason":    reason,
    }

    if os.path.exists(CORRECTIONS_LOG):
        df = pd.read_csv(CORRECTIONS_LOG)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(CORRECTIONS_LOG, index=False)


def get_accuracy_trend():
    """
    Returns the corrections log so the dashboard can show
    how MAE is improving over time.
    """
    if not os.path.exists(CORRECTIONS_LOG):
        return pd.DataFrame()
    return pd.read_csv(CORRECTIONS_LOG)


if __name__ == "__main__":
    # simulate a manager correction
    apply_correction(
        date="2025-04-28",
        day_of_week=0,
        is_raining=True,
        is_holiday=False,
        is_event=False,
        predicted_covers=120,
        actual_covers=85,
        reason="Heavy rain kept people away",
    )

    df = get_accuracy_trend()
    print(df.tail())