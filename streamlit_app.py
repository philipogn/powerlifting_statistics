import streamlit as st
import joblib
import pandas as pd
import sys
from pathlib import Path

from src.inference_service import predict_from_meets
from api.scraper import MeetScraper

DEFAULT_MODEL_PATH = 'models/XGBR_model_v1.pkl'

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

def main():
    # INFO/HEADERS
    st.set_page_config(page_title="Powerlifting Total Predictor", layout="centered")
    st.title("Powerlifting Predictor")
    st.caption("Predict the next meet TotalKg for an OpenPowerlifting lifter.")

    # MAIN PAGE
    username = st.text_input(f"OpenPowerlifting Username", placeholder="e.g. russelorhii")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Current age", min_value=14, max_value=80, value=None)
    with col2:
        bodyweight = st.number_input("Current bodyweight", min_value=30.0, max_value=350.0, value=None)
    
    button_action = st.button("Submit")
    if not button_action:
        return

    if not username:
        st.error("Enter a username")
        return
    if not age or not bodyweight:
        st.error("Age and Bodyweight field required for prediction")
        return

    # LOAD MODEL
    try:
        loaded_model = load_model(DEFAULT_MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model from {DEFAULT_MODEL_PATH}: {e}")

    with st.spinner("Fetching lifting history..."):
        try:
            pulled_data = MeetScraper(username=username)
            data = pulled_data.get_lifter_history()
            lifter = data.lifter
            meets = data.meet_details
        except Exception:
            st.error(f"Failed to fetch lifter history. Please try again")
            return

    if not meets:
        st.error("No competition history found")
        return
    if len(meets) < 2:
        st.error("At least 2 competitions are required to build history features.")
        return
    
    meets_df = pd.DataFrame(meets)

    # PREDICT 
    try:
        prediction, current_total, features = predict_from_meets(
            model = loaded_model,
            meets_df = meets_df,
            age = int(age),
            bodyweight = float(bodyweight),
            sex = lifter["Sex"]
        )
    except Exception as e:
        st.error(f"{e}")
        return
    
    improvement_kg = round(prediction - current_total, 2) if current_total else None

    met1, met2, met3 = st.columns(3)
    met1.metric("Predicted Total", f"{prediction:.2f} Kg", )
    met2.metric("Current Total", f"{current_total:.2f} Kg")
    met3.metric("Improvement Potential", f"{improvement_kg:.2f} Kg")

    with st.expander("Model inputs used", expanded=False):
        st.json(
            {
                "sex": lifter["Sex"],
                "age": age,
                "bodyweight_kg": bodyweight,
                **{k: v for k, v in features.items()}
            }
        )
    st.divider()
    st.header("Competition History")
    st.dataframe(meets)

if __name__ == "__main__":
    main()
