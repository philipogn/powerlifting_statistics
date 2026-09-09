import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import joblib
import yaml
import json

class TrainingPipeline():
    def __init__(self, config_path: str='config/local.yaml'):
        self.config = yaml.safe_load(open(config_path))
        self.pipeline = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.residual_quantiles = None

    def build_pipeline(self, feature_cols):
        '''
        Pipeline for imputation and encoding
        '''
        model_config = self.config['model']['xgboost']

        preprocessor = ColumnTransformer(
            transformers=[
                ('imputer', SimpleImputer(strategy='mean'), feature_cols),
                ('sex_encoder', OrdinalEncoder(), ['Sex'])
            ], remainder='drop'
        )
        return Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(**model_config))
        ])

    @staticmethod
    def print_metrics(label, y_true, y_pred):
        print(f'========= {label} RESULTS =========')
        print(f'Mean Absolute Error: {mean_absolute_error(y_true, y_pred):.4f}')
        print(f'Root Mean Squared Error: {root_mean_squared_error(y_true, y_pred):.4f}')
        print(f'R2 Score: {r2_score(y_true, y_pred):.4f}')
        print(f'====================================')

    def train_from_data(self, df):
        feature_cols = self.config['features']['columns']
        split_config = self.config['train_test_split']

        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])

        val_date = df['Date'].quantile(split_config['validation_quantile'])
        test_date = df['Date'].quantile(split_config['test_quantile'])
        train_df = df[df['Date'] < val_date]
        val_df = df[(df['Date'] >= val_date) & (df['Date'] < test_date)]
        test_df = df[df['Date'] > test_date]
        self.train_df, self.val_df, self.test_df = train_df, val_df, test_df

        cols = feature_cols + ['Sex']

        val_pipeline = self.build_pipeline(feature_cols)
        val_pipeline.fit(train_df[cols], train_df['TotalKg'])
        val_pred = val_pipeline.predict(val_df[cols])
        self.print_metrics('VALIDATION', val_df['TotalKg'], val_pred)

        # out-of-sample residual spread -> prediction intervals at serving time
        # (cheap global interval)
        residuals = val_df['TotalKg'].to_numpy() - val_pred
        self.residual_quantiles = {
            'q10': float(np.quantile(residuals, 0.1)),
            'q90': float(np.quantile(residuals, 0.9))
        }

        pipeline = self.build_pipeline(feature_cols)
        pipeline.fit(pd.concat([train_df, val_df])[cols], pd.concat([train_df, val_df])['TotalKg'])
        test_pred = pipeline.predict(test_df[cols])
        self.pipeline = pipeline
        self.print_metrics('TEST', test_df['TotalKg'], test_pred)
        return pipeline

    def save_model(self, save_path='models/XGBR_model_v1.pkl'):
        joblib.dump(self.pipeline, save_path)

    def save_intervals(self, save_path='models/prediction_intervals.json'):
        '''
        residual quantiles from the validation fold, 
        to show a prediction range instead of a bare point estimate for streamlit
        '''
        with open(save_path, 'w') as f:
            json.dump(self.residual_quantiles, f)
        print(f'Prediction intervals saved to "{save_path}"')

if __name__ == '__main__':
    df = pd.read_csv('data/3-features/opl_features_IPF.csv')
    train = TrainingPipeline()
    train.train_from_data(df)