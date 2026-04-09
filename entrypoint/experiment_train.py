import argparse
import sys
import os
import joblib
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_cleaning import DataProcessor
from feature_engineering import FeatureEngineering
from training import TrainingPipeline

def _validate_input_csv(raw_data_path):
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f'CSV not found: {raw_data_path}')
    if not raw_data_path.lower().endswith('.csv'):
        raise ValueError(f'Input must be a CSV file')

def _timestamped_path(path):
    stem, ext = os.path.splitext(path)
    if not ext:
        ext = '.csv'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{stem}_{timestamp}{ext}'


def preprocess_step(raw_data_path, preprocessed_output_path=None):
    print(f'[1/4] Preprocessing raw data from: {raw_data_path}')
    preprocess = DataProcessor(
        save_path=preprocessed_output_path,
        save_to_csv=True
    )
    data = preprocess.transform(raw_data_path)
    print(f'[1/4] Preprocessing complete. Rows: {len(data):,}')
    return data

def training_step(data, features_output_path, model_output_path):
    print(f'[2/4] Engineering features')
    features = FeatureEngineering(
        save_path=features_output_path,
        save_to_csv=True
    )
    engineered_features = features.engineer_features(data)
    print(f'[2/4] Feature engineering complete. Rows: {len(engineered_features):,}')

    print(f'[3/4] Training model')
    train = TrainingPipeline()
    train.train_from_data(engineered_features)
    print(f'[4/4] Saving model to: {model_output_path}')
    joblib.dump(train.pipeline, model_output_path)
    print(f'[4/4] Model saved successfully')

def run_entire_pipeline(raw_data_path, preprocessed_output_path, features_output_path, model_output_path):
    _validate_input_csv(raw_data_path)
    data = preprocess_step(raw_data_path, preprocessed_output_path)
    training_step(data, features_output_path, model_output_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run full powerlifting training pipeline from a CSV file.'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to raw input CSV file.'
    )
    parser.add_argument(
        '--preprocessed-output',
        default='data/2-preprocessed/opl_preprocessed_IPF.csv',
        help='Optional output path for preprocessed CSV.'
    )
    parser.add_argument(
        '--features-output',
        default='data/3-features/opl_features_IPF.csv',
        help='Optional output path for engineered features CSV.'
    )
    parser.add_argument(
        '--model-output',
        default='models/XGBR_model.pkl',
        help='Output path for trained model file.'
    )
    parser.add_argument(
        '--version-outputs',
        action='store_true',
        help='Append a timestamp to output filenames to avoid overwriting previous runs.'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    preprocessed_output = _timestamped_path(args.preprocessed_output) if args.version_outputs else args.preprocessed_output
    features_output = _timestamped_path(args.features_output) if args.version_outputs else args.features_output
    model_output = _timestamped_path(args.model_output) if args.version_outputs else args.model_output
    
    try:
        run_entire_pipeline(
            raw_data_path = args.input,
            preprocessed_output_path = preprocessed_output,
            features_output_path = features_output,
            model_output_path = model_output
        )
    except Exception as e:
        raise(f'Pipeline failed: {e}')