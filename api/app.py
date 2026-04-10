from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from scraper import MeetScraper
from src.inference_service import predict_from_meets

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


# ===== SCHEMAS =====
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
        
        prediction, current_total, features = predict_from_meets(
            model=model,
            meets_df=df,
            age=request.age,
            bodyweight=request.bodyweight,
            sex=lifter['Sex'],
        )
        improvement_kg = round(prediction - current_total, 2) if current_total else None
        
        return {
            'predicted_total_kg': round(prediction, 2),
            'current_total_kg': current_total,
            'improvement_potential_kg': improvement_kg,
            'lifter_profile': {
                'name': request.username,
                'bodyweight_kg': latest_meet['Weight'],
                'age': latest_meet['Age'],
                'latest_competition_date': latest_meet.get('Date')
            },
            'competition_history_count': len(meets),
            'features_used': {
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
            'competition_count': len(meets.meet_details),
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