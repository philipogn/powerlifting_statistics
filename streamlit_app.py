import streamlit as st
import joblib
import sys
from pathlib import Path

from src.inference_service import predict_from_meets
from api.scraper import MeetScraper

DEFAULT_MODEL_PATH = 'models/XGBR_model_v1.pkl'

def load_model(model_path):
    return joblib.load(model_path)

def configure_page():
    # INFO/HEADERS
    st.set_page_config(page_title="Powerlifting Total Predictor", layout="wide")
    st.title("Powerlifting Predictor")
    st.caption("Predict the next meet TotalKg for an OpenPowerlifting lifter.")

    # MAIN PAGE
    username = st.text_input(f"OpenPowerlifting Username", placeholder="e.g. russelorhii")
    st.number_input("Current age", min_value=14, max_value=80, value=None)
    st.number_input("Current bodyweight", min_value=30.0, max_value=350.0, value=None)
    st.button("Submit")

    if not username:
        st.error("Enter a username")
        return

    # LOAD MODEL
    try:
        model = load_model(DEFAULT_MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model from {DEFAULT_MODEL_PATH}: {e}")

    try:
        pulled_data = MeetScraper(username=username)
        data = pulled_data.get_lifter_history()
        lifter = data.lifter
        meets = data.meet_details
    except Exception:
        st.error(f"Failed to fetch lifter history. Please try again")

    st.header("Competition History")
    st.dataframe(meets)


    # try:
    #     predict_from_meets(
    #         model = mode
    #     )

def main():
    configure_page()

if __name__ == "__main__":
    main()

'''
username input
age, bodyweight input
predicted, current, improvement in kgs
list comp history?
'''