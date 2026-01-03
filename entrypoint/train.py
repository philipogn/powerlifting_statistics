import yaml
import pandas as pd
import sys
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_cleaning import DataProcessor
from feature_engineering import FeatureEngineering
from training import TrainingPipeline

config = yaml.safe_load(open('config/prod.yaml'))

def run_entire_pipeline():
    try:
        clean_df = DataProcessor.run(config['data']['raw_data_path'])
        feat_eng = FeatureEngineering.run(clean_df)
        train = TrainingPipeline.run(feat_eng)
    except:
        print('Error during training pipeline')

def predict_on_unseen():
    test_df = pd.read_csv('data/3-features/Test_dataset.csv')
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    true_values = test_df['TotalKg']
    # need to clean up dataset probably
    test_df = test_df.drop(columns=['Name', 'Date', 'Sex', 'Division', 'Age', 'Best3BenchKg', 'Best3SquatKg', 'Best3DeadliftKg', 'TotalKg', 'BodyweightKg'])
    try:
        loaded_model = joblib.load('models/XGBR_model.pkl')
        predictions = loaded_model.predict(test_df)
        mae = mean_absolute_error(true_values, predictions)
        rmse = root_mean_squared_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        print(f'MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}')
    except Exception as e:
        print(e)




if __name__ == '__main__':
    # run_entire_pipeline()
    predict_on_unseen()