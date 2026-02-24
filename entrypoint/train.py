import yaml
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_cleaning import DataProcessor
from feature_engineering import FeatureEngineering
from training import TrainingPipeline

config = yaml.safe_load(open('config/prod.yaml'))

def preprocess_step(raw_data_path):
    try:
        # raw_data = pd.read_csv(
        #     raw_data_path, 
        #     dtype={'Tested': 'string', 'State': 'string', 'ParentFederation': 'string', 'MeetState': 'string'})
        preprocess = DataProcessor(
            save_path='data/2-preprocessed/opl_preprocessed_IPF.csv',
            save_to_csv=True
        )
        preprocess.transform(raw_data_path)
    except FileNotFoundError as e:
        print(f'Error loading raw dataset: {e}')

def training_step():
    try:
        df = pd.read_csv('data/2-preprocessed/opl_preprocessed_IPF.csv')
        features = FeatureEngineering(
            save_path='data/3-features/opl_features_IPF.csv', 
            save_to_csv=True
        )
        features.engineer_features(df)

        df = pd.read_csv('data/3-features/opl_features_IPF.csv')
        train = TrainingPipeline()
        train.train_from_data(df)
        train.save_model()
    except FileNotFoundError as e:
        print(f'Preprocessed dataset not found: {e}')

# save model

''' 
MODIFY SO DATA CSV CAN BE DIRECTLY INJCECTED FROM TERMINAL?
e.g., python src/entrypoint --input {csv_file}.csv
'''

def run_entire_pipeline(raw_data_path):
    preprocess_step(raw_data_path)
    training_step()

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
    run_entire_pipeline(raw_data_path='data/1-raw/openpowerlifting-2025-09-27.csv')
    # predict_on_unseen()