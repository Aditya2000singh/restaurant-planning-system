import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, "src")
from predictor import predict_covers, predict_staff, predict_ingredients, train_model
from feedback  import apply_correction, get_accuracy_trend

st.set_page_config(page_title="Restaurant RPS", page_icon="🍽️", layout="wide")
st.title("🍽️ Restaurant Resource Planning System")


# ---------- sidebar: day setup ----------

st.sidebar.header("Set up tomorrow")
day_names   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
day_of_week = st.sidebar.selectbox("Day of week", range(7), format_func=lambda x: day_names[x])
is_raining  = st.sidebar.checkbox("Rain forecast")
is_holiday  = st.sidebar.checkbox("Public holiday")
is_event    = st.sidebar.checkbox("Local event nearby")

if st.sidebar.button("Get Prediction"):
    covers      = predict_covers(day_of_week, is_raining, is_holiday, is_event)
    staff       = predict_staff(covers)
    ingredients = predict_ingredients(covers)

    st.session_state["last_prediction"] = covers

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Covers", covers)

    with col2:
        st.subheader("Staff Schedule")
        st.write(f"👨‍🍳 Kitchen: **{staff['kitchen_staff']}** people")
        st.write(f"🧑‍🍽️ Floor:   **{staff['floor_staff']}** people")
        st.write(f"🍸 Bar:     **{staff['bar_staff']}** people")

    with col3:
        st.subheader("Ingredient Order")
        st.write(f"🐔 Chicken: **{ingredients['chicken_kg']} kg**")
        st.write(f"🥦 Veggies: **{ingredients['veggies_kg']} kg**")
        st.write(f"🌾 Flour:   **{ingredients['flour_kg']} kg**")


st.divider()


# ---------- manager correction section ----------

st.subheader("📝 Submit Actual Covers (Manager Correction)")
st.write("After service ends, enter what actually happened. The model will learn from it.")

with st.form("correction_form"):
    corr_date      = st.date_input("Date of service")
    corr_predicted = st.number_input("Predicted covers", min_value=0, value=st.session_state.get("last_prediction", 100))
    corr_actual    = st.number_input("Actual covers",    min_value=0, value=100)
    corr_rain      = st.checkbox("It was raining")
    corr_holiday   = st.checkbox("It was a holiday")
    corr_event     = st.checkbox("There was a local event")
    corr_reason    = st.text_input("Why was it different? (optional)", placeholder="e.g. heavy rain, big match on TV")
    submitted      = st.form_submit_button("Submit Correction")

    if submitted:
        apply_correction(
            date=str(corr_date),
            day_of_week=corr_date.weekday(),
            is_raining=corr_rain,
            is_holiday=corr_holiday,
            is_event=corr_event,
            predicted_covers=int(corr_predicted),
            actual_covers=int(corr_actual),
            reason=corr_reason,
        )
        st.success("✅ Model updated! It will be more accurate next time.")


st.divider()


# ---------- accuracy trend chart ----------

st.subheader("📈 Model Learning Over Time")
df = get_accuracy_trend()

if df.empty:
    st.info("No corrections yet — submit some above to see the learning curve.")
else:
    st.write(f"Total corrections: **{len(df)}** | "
             f"Average error: **{df['error'].mean():.1f} covers** | "
             f"Latest error: **{df['error'].iloc[-1]} covers**")
    st.line_chart(df[["date","pct_error"]].set_index("date"))


st.divider()


# ---------- retrain from scratch ----------

st.subheader("🔄 Retrain Model from Scratch")
st.write("If you've added new historical data, retrain the base model here.")
if st.button("Retrain"):
    train_model()
    st.success("Model retrained on historical data.")