import datetime
import pandas as pd
from typing import Tuple


def get_max_lifts(history_meet: pd.DataFrame) -> pd.DataFrame:
    history_processed = []
    for _, meet in history_meet.iterrows():
        squat = max(meet["Squat"]) if meet["Squat"] else 0
        bench = max(meet["Bench"]) if meet["Bench"] else 0
        deadlift = max(meet["Deadlift"]) if meet["Deadlift"] else 0

        if squat and bench and deadlift:
            history_processed.append(
                {
                    "Best3SquatKg": squat,
                    "Best3BenchKg": bench,
                    "Best3DeadliftKg": deadlift,
                    "TotalKg": float(meet.get("Total", 0) or 0),
                    "Date": meet.get("Date", None),
                }
            )
        else:
            continue
    return pd.DataFrame(history_processed)


def create_features_from_history(history_df: pd.DataFrame) -> dict:
    features = {}
    features["prev_squat"] = history_df["Best3SquatKg"].iloc[-1]
    features["prev_bench"] = history_df["Best3BenchKg"].iloc[-1]
    features["prev_deadlift"] = history_df["Best3DeadliftKg"].iloc[-1]

    features["avg_squat"] = history_df["Best3SquatKg"].mean()
    features["avg_bench"] = history_df["Best3BenchKg"].mean()
    features["avg_deadlift"] = history_df["Best3DeadliftKg"].mean()

    features["days_since_last_meet"] = (
        pd.to_datetime(datetime.datetime.today()) - pd.to_datetime(history_df["Date"].iloc[-1])
    ).days
    features["total_meets"] = len(history_df)

    if len(history_df) >= 2:
        first, last, second_last = history_df["TotalKg"].iloc[0], history_df["TotalKg"].iloc[-1], history_df["TotalKg"].iloc[-2]
        features["percent_gain_since_last"] = (last - second_last) / second_last if second_last else 0
        features["career_avg_improvement_rate"] = ((last - first) / first / (len(history_df) - 1) if first else 0)
    else:
        features["percent_gain_since_last"] = 0
        features["career_avg_improvement_rate"] = 0

    features["total_std"] = history_df["TotalKg"].std() if len(history_df) > 1 else 0

    return features


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
    # latest_meet = meets_df.iloc[-1]
    history_df = get_max_lifts(meets_df)
    if len(history_df) < 2:
        raise ValueError("At least 2 valid meets are required to generate features.")

    features = create_features_from_history(history_df)
    X = prepare_model_input(features, age=age, bodyweight=bodyweight, sex=sex)
    prediction = float(model.predict(X)[0])
    current_total = float(meets_df.iloc[-1].get("Total", 0) or 0)
    return prediction, current_total, features
