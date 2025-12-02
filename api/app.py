from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from scraper import MeetScraper

app = FastAPI(
    title="Powerlifting Total Predictor API",
    description="Predicts next total based on competition history",
    version="1.0.0"
)

MODEL_PATH = Path(__file__).parent.parent / "models" / "XGBR_model.pkl"
try:
    with open(MODEL_PATH, 'rb') as f:
        model = joblib.load(f)
    print(f"Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None



# SCHEMAS
class UsernameRequest(BaseModel):
    username: str = Field(max_length=50, description="OpenPowerlifting username")

class ProfilePredictionResponse(BaseModel):
    predicted_total_kg: float
    current_total_kg: float = None
    improvement_potential_kg: float = None
    lifter_profile: dict
    competition_history_count: int
    features_used: dict




# FEATURE ENGINEERING FUNCTIONS

def create_features_from_history(current_data, history_df: pd.DataFrame):
    features = current_data.copy()
    
    features['prev_squat'] = history_df['Best3SquatKg'].iloc[-1]
    features['prev_bench'] = history_df['Best3BenchKg'].iloc[-1]
    features['prev_deadlift'] = history_df['Best3DeadliftKg'].iloc[-1]
    features['avg_squat'] = history_df['Best3SquatKg'].mean()
    features['avg_bench'] = history_df['Best3BenchKg'].mean()
    features['avg_deadlift'] = history_df['Best3DeadliftKg'].mean()

    features['bodyweight_change'] = history_df['BodyweightKg'].iloc[-1] - history_df['BodyweightKg'].iloc[-2]
    
    if len(history_df) >= 2:
        features['percent_gain_since_last'] = (
            history_df['TotalKg'].iloc[-1] - history_df['TotalKg'].iloc[-2]
        ) / history_df['TotalKg'].iloc[-2]
    else:
        features['percent_gain_since_last'] = 0
    
    return features

def prepare_model_input(features: dict) -> np.array:
    # create feature array in matching order with training data
    feature_values = [
        features.get('prev_squat', 0),
        features.get('prev_bench', 0),
        features.get('prev_deadlift', 0),
        features.get('avg_squat', 0),
        features.get('avg_bench', 0),
        features.get('avg_deadlift', 0),
        features.get('BodyweightKg', 0),
        features.get('bodyweight_change', 0),
        features.get('percent_gain_since_last', 0)
    ]
    return np.array([feature_values])



# ===== ENDPOINTS =====

@app.get("/")
def root():
    return {
        "message": "Powerlifting Total Predictor API",
        "description": "Predicts next competition total based on training history",
        "note": "Requires at least 1 previous competition for predictions",
        "endpoints": {
            "/predict-from-profile": "POST - Predict from OpenPowerlifting username",
        }
    }

@app.post('/predict_from_profile', response_model=ProfilePredictionResponse)
def predict_from_openpowerlifting(request: UsernameRequest):
    """
    Fetch lifter data from OpenPowerlifting and predict next total.
    """
    try:
        scrape = MeetScraper(username=request.username)
        meets = scrape.get_lifter_history()

        if not meets or len(meets) < 1:
            raise HTTPException(
                status_code=404,
                detail=f'No competition history found for {request.username}'
            )

        df = pd.DataFrame(meets)

        latest_meet = df.iloc[-1]
        history_meet = df.iloc[:]

        if len(history_meet) < 1:
            raise HTTPException(
                status_code=400,
                detail=f'{request.username} needs at least 2 competitions for prediction'
            )
        
        history_processed = []
        for index, meet in history_meet.iterrows():
            squat = max(meet['Squat']) if meet['Squat'] else 0
            bench = max(meet['Bench']) if meet['Bench'] else 0
            deadlift = max(meet['Deadlift']) if meet['Deadlift'] else 0

            if not squat or not bench or not deadlift:
                continue
            else:
                history_processed.append({
                    'Best3SquatKg': squat,
                    'Best3BenchKg': bench,
                    'Best3DeadliftKg': deadlift,
                    'TotalKg': float(meet.get('Total', 0)),
                    'BodyweightKg': float(meet.get('Weight', 0))
                    }
                )
        history_df = pd.DataFrame(history_processed)

        current_data = {
            'BodyweightKg': float(latest_meet.get('Weight', 0)),
            'Age': str(latest_meet.get('Age')),
            'Sex': latest_meet.get('Sex', 'M'),
        }

        features = create_features_from_history(current_data, history_df)
        X = prepare_model_input(features)
        print("FEATURES:", features)
        print("MODEL INPUT:", X)
        prediction = model.predict(X)

        current_total = float(latest_meet.get('Total', 0))
        improvement_kg = round(float(prediction) - current_total, 2) if current_total else None

        return {
            'predicted_total_kg': round(float(prediction), 2),
            'current_total_kg': current_total,
            'improvement_potential_kg': improvement_kg,
            'lifter_profile': {
                'name': request.username,
                'bodyweight_kg': current_data['BodyweightKg'],
                'age': current_data['Age'],
                'sex': current_data['Sex'],
                'lastest_competition_date': latest_meet.get('Date')
            },
            'competition_history_count': len(meets),
            "features_used": {
                "prev_squat": features['prev_squat'],
                "prev_bench": features['prev_bench'],
                "prev_deadlift": features['prev_deadlift'],
                "avg_squat": round(features['avg_squat'], 2),
                "avg_bench": round(features['avg_bench'], 2),
                "avg_deadlift": round(features['avg_deadlift'], 2),
                "bodyweight_kg": features['BodyweightKg'],
                "bodyweight_change": round(features['bodyweight_change'], 2),
                "percent_gain_since_last": round(features['percent_gain_since_last'], 4)
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
