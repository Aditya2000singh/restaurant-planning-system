from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date
import sys
import os

# make sure src/ is in path when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from predictor import predict_covers, predict_staff, predict_ingredients
from feedback  import apply_correction, get_accuracy_trend

app = FastAPI(title="Restaurant RPS", version="1.0")


# ---------- request/response models ----------

class PredictRequest(BaseModel):
    day_of_week: int          # 0=Monday ... 6=Sunday
    is_raining:  bool = False
    is_holiday:  bool = False
    is_event:    bool = False

class CorrectionRequest(BaseModel):
    date:              str
    day_of_week:       int
    is_raining:        bool = False
    is_holiday:        bool = False
    is_event:          bool = False
    predicted_covers:  int
    actual_covers:     int
    reason:            str = ""


# ---------- endpoints ----------

@app.get("/")
def home():
    return {"message": "Restaurant RPS API is running"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Given day info, returns predicted covers, staff schedule,
    and ingredient order list.
    """
    covers = predict_covers(
        day_of_week=req.day_of_week,
        is_raining=req.is_raining,
        is_holiday=req.is_holiday,
        is_event=req.is_event,
    )
    staff       = predict_staff(covers)
    ingredients = predict_ingredients(covers)

    return {
        "predicted_covers": covers,
        "staff_schedule":   staff,
        "ingredient_order": ingredients,
    }


@app.post("/correct")
def correct(req: CorrectionRequest):
    """
    Manager submits actual covers after the day.
    This triggers partial_fit() so the model learns from the mistake.
    """
    try:
        apply_correction(
            date=req.date,
            day_of_week=req.day_of_week,
            is_raining=req.is_raining,
            is_holiday=req.is_holiday,
            is_event=req.is_event,
            predicted_covers=req.predicted_covers,
            actual_covers=req.actual_covers,
            reason=req.reason,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Correction applied and model updated."}


@app.get("/accuracy")
def accuracy():
    """Returns the full log of corrections and error metrics."""
    df = get_accuracy_trend()
    if df.empty:
        return {"corrections": [], "message": "No corrections yet."}
    return {"corrections": df.to_dict(orient="records")}