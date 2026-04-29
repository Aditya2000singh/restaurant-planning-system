import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# paths where we save the trained model
MODEL_PATH  = "models/covers_model.pkl"
SCALER_PATH = "models/scaler.pkl"


def get_features(day_of_week, is_raining, is_holiday, is_event):
    """
    Turns raw inputs into the feature array the model expects.
    Nothing fancy — just a plain list.
    """
    return np.array([[
        day_of_week,
        int(is_raining),
        int(is_holiday),
        int(is_event),
        int(day_of_week >= 5),   # is_weekend flag
    ]])


def train_model(data_path="data/historical_data.csv"):
    """
    Trains a simple regression model on historical data.
    SGDRegressor supports partial_fit() which is what makes
    the self-learning part work later on.
    """
    df = pd.read_csv(data_path)

    features = df[["day_of_week", "is_raining", "is_holiday", "is_event"]].copy()
    features["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    X = features.values
    y = df["actual_covers"].values

    # scale so SGD trains better
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SGDRegressor(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    # save both so we can reload later
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Model trained and saved.")
    return model, scaler


def load_model():
    """Loads saved model + scaler from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No trained model found. Run train_model() first.")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_covers(day_of_week, is_raining=False, is_holiday=False, is_event=False):
    """Returns predicted cover count for a given day."""
    model, scaler = load_model()
    X = get_features(day_of_week, is_raining, is_holiday, is_event)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    return max(int(round(prediction)), 10)  # never predict below 10


def predict_staff(covers):
    """
    Simple rule-based staff calculation from cover count.
    Easier to explain and tweak than another ML model.
    """
    return {
        "kitchen_staff": max(2, covers // 25),
        "floor_staff":   max(2, covers // 20),
        "bar_staff":     max(1, covers // 50),
    }


def predict_ingredients(covers, lead_time_days=2, safety_buffer=1.15):
    """
    Ingredient order quantities based on predicted covers.
    safety_buffer = 1.15 means we order 15% extra just in case.
    lead_time_days is how many days ahead we need to order.
    """
    # per-cover usage in kg (learned from historical averages)
    per_cover = {
        "chicken_kg": 0.18,
        "veggies_kg": 0.12,
        "flour_kg":   0.08,
    }

    # shelf life in days so we know max we can hold
    shelf_life = {
        "chicken_kg": 3,
        "veggies_kg": 5,
        "flour_kg":   30,
    }

    order = {}
    for ingredient, usage in per_cover.items():
        raw_qty = covers * usage * safety_buffer
        # don't order more than what we can use before it expires
        max_can_hold = shelf_life[ingredient] * covers * usage
        order[ingredient] = round(min(raw_qty, max_can_hold), 2)

    return order


if __name__ == "__main__":
    # quick test
    train_model()
    covers = predict_covers(day_of_week=5, is_event=True)
    print(f"Predicted covers: {covers}")
    print(f"Staff needed: {predict_staff(covers)}")
    print(f"Ingredients to order: {predict_ingredients(covers)}")