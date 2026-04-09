from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import datetime

sys.path.append(str(Path(__file__).parent.parent))

from scraper import MeetScraper

app = FastAPI(
    title="Powerlifting Total Predictor API",
    description="Predicts next total based on competition history",
    version="1.0.0"
)

MODEL_PATH = Path(__file__).parent.parent / "models" / "XGBR_model_v1.pkl"
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
    age: int = Field(description="Current age")
    bodyweight: float = Field(description="Current bodyweight (KG)")

class ProfilePredictionResponse(BaseModel):
    predicted_total_kg: float
    current_total_kg: float = None
    improvement_potential_kg: float = None
    lifter_profile: dict
    competition_history_count: int
    features_used: dict




# FEATURE ENGINEERING FUNCTIONS

def create_features_from_history(history_df: pd.DataFrame):
    features = {}
    
    features['prev_squat'] = history_df['Best3SquatKg'].iloc[-1]
    features['prev_bench'] = history_df['Best3BenchKg'].iloc[-1]
    features['prev_deadlift'] = history_df['Best3DeadliftKg'].iloc[-1]
    
    features['avg_squat'] = history_df['Best3SquatKg'].mean()
    features['avg_bench'] = history_df['Best3BenchKg'].mean()
    features['avg_deadlift'] = history_df['Best3DeadliftKg'].mean()
    
    # need to change to get current date instead for actual prediction
    features['days_since_last_meet'] = (
        pd.to_datetime(datetime.datetime.today()) - pd.to_datetime(history_df['Date'].iloc[-1])
    ).days
    
    features['total_meets'] = len(history_df)
    
    if len(history_df) >= 2:
        first, last, second_last = history_df['TotalKg'].iloc[0], history_df['TotalKg'].iloc[-1], history_df['TotalKg'].iloc[-2]
        features['percent_gain_since_last'] = ((last - second_last) / second_last)
        features['career_avg_improvement_rate'] = ((last - first) / first / (len(history_df) - 1))
    else:
        features['percent_gain_since_last'] = 0
        features['career_avg_improvement_rate'] = 0
    
    features['total_std'] = history_df['TotalKg'].std() if len(history_df) > 1 else 0
    
    return features


def prepare_model_input(features: dict, age, bodyweight, sex):
    # create feature dict in matching order with training data
    feature_values = {
        'Age': age, 
        'BodyweightKg': bodyweight,
        'prev_squat': features.get('prev_squat', 0),
        'prev_bench': features.get('prev_bench', 0),
        'prev_deadlift': features.get('prev_deadlift', 0),
        'avg_squat': features.get('avg_squat', 0),
        'avg_bench': features.get('avg_bench', 0),
        'avg_deadlift': features.get('avg_deadlift', 0),
        'days_since_last_meet': features.get('days_since_last_meet', 0),
        'total_meets': features.get('total_meets', 0),
        'percent_gain_since_last': features.get('percent_gain_since_last', 0),
        'career_avg_improvement_rate': features.get('career_avg_improvement_rate', 0),
        'total_std': features.get('total_std', 0),
        'Sex': sex
    }
    return pd.DataFrame([feature_values])

def get_max_lifts(history_meet):
    history_processed = []
    for index, meet in history_meet.iterrows():
        squat = max(meet['Squat']) if meet['Squat'] else 0
        bench = max(meet['Bench']) if meet['Bench'] else 0
        deadlift = max(meet['Deadlift']) if meet['Deadlift'] else 0
        
        if squat and bench and deadlift:
            history_processed.append({
                'Best3SquatKg': squat,
                'Best3BenchKg': bench,
                'Best3DeadliftKg': deadlift,
                'TotalKg': float(meet.get('Total', 0)),
                'Date': meet.get('Date', None)
                }
            )
        else:
            continue
    # return history_processed
    return pd.DataFrame(history_processed)



# ===== ENDPOINTS =====

@app.get("/")
def root():
    return {
        "message": "Powerlifting Total Predictor API",
        "description": "Predicts next competition total based on training history",
        "note": "Requires at least 1 previous competition for predictions",
        "endpoints": {
            "/predictions": "POST - Predict from OpenPowerlifting username",
            "/competitions/{username}": "GET - View competition history of lifter",
            "/heath": "Health status of model",
            "/docs": "Interactive API with SwaggerUI"
        }
    }

@app.post('/predictions', response_model=ProfilePredictionResponse)
def predict_from_openpowerlifting(request: UsernameRequest):
    """
    Fetch lifter data from OpenPowerlifting and predict next total.
    """
    try:
        scrape = MeetScraper(username=request.username)
        data = scrape.get_lifter_history()
        meets, lifter = data.meet_details, data.lifter

        if not meets or len(meets) < 1:
            raise HTTPException(
                status_code=404,
                detail=f'No competition history found for {request.username}'
            )
        elif len(meets) < 2:
            raise HTTPException(
                status_code=400,
                detail=f'{request.username} needs at least 2 competitions for predictions'
            )
        
        df = pd.DataFrame(meets)
        
        latest_meet = df.iloc[-1]
        history_df = get_max_lifts(df)
        features = create_features_from_history(history_df)

        current_age = request.age
        current_bodyweight = request.bodyweight
        sex = lifter['Sex']
        X = prepare_model_input(features, current_age, current_bodyweight, sex)

        prediction = model.predict(X)
        
        current_total = float(latest_meet.get('Total', 0))
        improvement_kg = round(float(prediction) - current_total, 2) if current_total else None
        
        return {
            'predicted_total_kg': round(float(prediction), 2),
            'current_total_kg': current_total,
            'improvement_potential_kg': improvement_kg,
            'lifter_profile': {
                'name': request.username,
                'bodyweight_kg': latest_meet['Weight'],
                'age': latest_meet['Age'],
                'lastest_competition_date': latest_meet.get('Date')
            },
            'competition_history_count': len(meets),
            'features_used': { # add others
                'prev_squat': features['prev_squat'],
                'prev_bench': features['prev_bench'],
                'prev_deadlift': features['prev_deadlift'],
                'avg_squat': round(features['avg_squat'], 2),
                'avg_bench': round(features['avg_bench'], 2),
                'avg_deadlift': round(features['avg_deadlift'], 2),
                'days_since_last_meet': round(features['days_since_last_meet']),
                'total_meets': round(features['total_meets']),
                'percent_gain_since_last': round(features['percent_gain_since_last'], 4),
                'career_avg_improvement_rate': round(features['career_avg_improvement_rate']),
                'total_std': round(features['total_std'])
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get('/competitions/{name}')
def get_competition_history(name: str):
    '''
    Fetch competition history from OpenPowerlifting
    '''
    # ADD OPTION TO FILTER BASED ON EQUIPMENT (RAW, SINGLE, WRAPS...)
    try:
        scrape = MeetScraper(name)
        meets = scrape.get_lifter_history()
        if not meets:
            raise HTTPException(status_code=404, detail=f'No competition history for {name}')
        
        return {
            'name': name,
            'competition_count': len(meets),
            'competitions': meets
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get('/health')
def health_check():
    return {
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None
    }