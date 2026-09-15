import json
from pathlib import Path
import streamlit as st
import joblib
import pandas as pd

from src.inference_service import predict_from_meets, select_interval
from api.scraper import MeetScraper

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / 'models' / 'XGBR_model_v1.pkl'
INTERVALS_PATH = APP_DIR / 'models' / 'prediction_intervals.json'

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

@st.cache_resource
def load_intervals(intervals_path):
    ''' validation residual quantiles written by training '''
    try:
        with open(intervals_path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None

def main():
    # INFO/HEADERS
    st.set_page_config(page_title="Powerlifting Total Predictor", layout="centered")
    st.title("Powerlifting Total Predictor")
    st.caption("Predict the next meet TotalKg for an OpenPowerlifting lifter.")

    # MAIN PAGE
    username = st.text_input(f"OpenPowerlifting Username", placeholder="e.g. russelorhii")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Current age", min_value=14, max_value=80, value=None)
    with col2:
        bodyweight = st.number_input("Current bodyweight", min_value=30.0, max_value=350.0, value=None)
    
    button_action = st.button("Predict next total", type="primary")
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
        return

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
    percent_gain = round((improvement_kg / current_total) * 100, 2) if improvement_kg else None

    st.success("Prediction complete.")
    met1, met2, met3 = st.columns(3)
    met1.metric("Predicted Total", f"{prediction:.2f} Kg", f"{percent_gain}%")
    met2.metric("Current Total", f"{current_total:.2f} Kg")
    met3.metric("Improvement Potential", f"{improvement_kg:.2f} Kg")

    intervals = load_intervals(INTERVALS_PATH)
    if intervals:
        band = select_interval(intervals, features.get('days_since_last_meet'))
        low, high = prediction + band['q10'], prediction + band['q90']
        basis = f"lifters returning after {band['label']}" if band['label'] else "similar predictions"
        st.caption(f"Likely range: **{low:.1f} - {high:.1f} kg** (80% of {basis} land in this band).")
    st.caption(f"For reference, simply repeating the last total would predict {current_total:.1f} kg.")

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
