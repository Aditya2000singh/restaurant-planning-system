import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


random.seed(42)
np.random.seed(42)

def generate_data(days=180):
    """
    Creates fake historical restaurant data for the past N days.
    Each row = one day's summary.
    """
    rows = []
    start_date = datetime.today() - timedelta(days=days)

    for i in range(days):
        date = start_date + timedelta(days=i)
        day_of_week = date.weekday() 

        
        base_covers = 80 if day_of_week < 5 else 140


        is_raining = random.random() < 0.25
        is_holiday  = random.random() < 0.08
        is_event    = random.random() < 0.12

        covers = base_covers
        covers += 30 if is_holiday else 0
        covers += 40 if is_event   else 0
        covers -= 20 if is_raining else 0
        covers += random.randint(-15, 15)   
        covers = max(covers, 20)            


        kitchen_staff = max(2, covers // 25)
        floor_staff   = max(2, covers // 20)
        bar_staff     = max(1, covers // 50)

        
        chicken_kg = round(covers * 0.18 + random.uniform(-2, 2), 2)
        veggies_kg = round(covers * 0.12 + random.uniform(-1, 1), 2)
        flour_kg   = round(covers * 0.08 + random.uniform(-1, 1), 2)

        rows.append({
            "date":          date.strftime("%Y-%m-%d"),
            "day_of_week":   day_of_week,
            "is_raining":    int(is_raining),
            "is_holiday":    int(is_holiday),
            "is_event":      int(is_event),
            "actual_covers": covers,
            "kitchen_staff": kitchen_staff,
            "floor_staff":   floor_staff,
            "bar_staff":     bar_staff,
            "chicken_kg":    chicken_kg,
            "veggies_kg":    veggies_kg,
            "flour_kg":      flour_kg,
        })

    df = pd.DataFrame(rows)
    df.to_csv("data/historical_data.csv", index=False)
    print(f"Generated {len(df)} days of data -> data/historical_data.csv")
    return df


if __name__ == "__main__":
    generate_data()