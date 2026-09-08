import datetime
import sys
from pathlib import Path
from typing import Tuple
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import FeatureEngineering


HISTORY_COLUMNS = ['Date', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg']


def prepare_history(meets_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Meets arrive from api.scraper already filtered (Raw SBD, valid total, no DQ) and in the training schema.
    This just selects the feature-relevant columns and drops any meet with a missing lift value.
    '''
    return meets_df[HISTORY_COLUMNS].dropna()


def create_features_from_history(history_df: pd.DataFrame, as_of=None) -> dict:
    '''
    Uses the training-time feature builder so serving features can never drift from training. 
    as_of stands in for the (unknown) next meet date when computing days_since_last_meet (defaults to today)
    '''
    as_of = as_of if as_of is not None else datetime.datetime.today()
    return FeatureEngineering().create_features({'Date': as_of}, history_df)


def prepare_model_input(features: dict, age: int, bodyweight: float, sex: str) -> pd.DataFrame:
    features_values = {
        "Age": age,
        "BodyweightKg": bodyweight,
        "prev_squat": features.get("prev_squat", 0),
        "prev_bench": features.get("prev_bench", 0),
        "prev_deadlift": features.get("prev_deadlift", 0),
        "avg_squat": features.get("avg_squat", 0),
        "avg_bench": features.get("avg_bench", 0),
        "avg_deadlift": features.get("avg_deadlift", 0),
        "days_since_last_meet": features.get("days_since_last_meet", 0),
        "total_meets": features.get("total_meets", 0),
        "percent_gain_since_last": features.get("percent_gain_since_last", 0),
        "career_avg_improvement_rate": features.get("career_avg_improvement_rate", 0),
        "total_std": features.get("total_std", 0),
        "Sex": sex,
    }
    return pd.DataFrame([features_values])


def predict_from_meets(model, meets_df: pd.DataFrame, age: int, bodyweight: float, sex: str) -> Tuple[float, float, dict]:
    history_df = prepare_history(meets_df)
    if len(history_df) < 2:
        raise ValueError("At least 2 valid meets are required to generate features.")

    features = create_features_from_history(history_df)
    X = prepare_model_input(features, age=age, bodyweight=bodyweight, sex=sex)
    prediction = float(model.predict(X)[0])
    current_total = float(history_df["TotalKg"].iloc[-1])
    return prediction, current_total, features
