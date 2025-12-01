from fastapi import FastAPI
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

app = FastAPI(
    title="Powerlifting Total Predictor API",
    description="Predicts next total based on competition history",
    version="1.0.0"
)

MODEL_PATH = Path(__file__).parent.parent / "models" / "XGBR_model.pkl"
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

