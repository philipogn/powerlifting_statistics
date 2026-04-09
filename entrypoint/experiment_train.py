import yaml
import sys
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_cleaning import DataProcessor
from feature_engineering import FeatureEngineering
from training import TrainingPipeline

config = yaml.safe_load(open('config/prod.yaml'))

def preprocess_step(raw_data_path):
    try:
        preprocess = DataProcessor()
        data = preprocess.transform(raw_data_path)
    except FileNotFoundError as e:
        print(f'Error loading raw dataset: {e}')
    return data

def training_step(data):
    try:
        features = FeatureEngineering()
        engineered_features = features.engineer_features(data)

        train = TrainingPipeline()
        train.train_from_data(engineered_features)
    except FileNotFoundError as e:
        print(f'Preprocessed dataset not found: {e}')

def run_entire_pipeline(raw_data_path):
    data = preprocess_step(raw_data_path)
    training_step(data)

if __name__ == '__main__':
    run_entire_pipeline(raw_data_path='data/1-raw/openpowerlifting-2025-09-27.csv')
    # predict_on_unseen()