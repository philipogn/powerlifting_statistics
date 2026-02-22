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

def run_entire_pipeline(raw_data):
    try:
        preprocess = DataProcessor(
            save_path='data/2-preprocessed/openpowerlifting_preprocessed.csv',
            save_to_csv=True
        )
        preprocess.transform(raw_data)

        df = pd.read_csv('data/2-preprocessed/openpowerlifting_preprocessed.csv')
        features = FeatureEngineering(
            save_path='data/3-features/openpowerlifting_features.csv', 
            save_to_csv=True
        )
        features.engineer_features(df)

        train = TrainingPipeline.run()
        train.train_from_data(df)
    except:
        print('Error during training pipeline')

# def predict_on_unseen():
#     test_df = pd.read_csv('data/3-features/Test_dataset.csv')
#     test_df['Date'] = pd.to_datetime(test_df['Date'])
#     true_values = test_df['TotalKg']
#     '''
#     need to clean up dataset probably
#     just get feature columns from yaml 
#     (this will work for now, re-extract only engineered features later)
#     '''
#     test_df = test_df[config['features']['columns']]
#     try:
#         loaded_model = joblib.load('models/XGBR_model.pkl')
#         predictions = loaded_model.predict(test_df)
#         mae = mean_absolute_error(true_values, predictions)
#         rmse = root_mean_squared_error(true_values, predictions)
#         r2 = r2_score(true_values, predictions)
#         print(f'MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}')
#     except Exception as e:
#         print(e)

if __name__ == '__main__':
    raw_data = pd.read_csv('data/1-raw/openpowerlifting-2025-09-27.csv')
    run_entire_pipeline(raw_data)
    # predict_on_unseen()