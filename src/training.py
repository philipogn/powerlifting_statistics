import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import joblib
import yaml
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import Evaluator


class TrainingPipeline():
    def __init__(self, config_path: str='config/local.yaml'):
        self.config = yaml.safe_load(open(config_path))
        self.pipeline = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.residual_quantiles = None
        self.val_pred = None
        self.val_residuals = None

    def build_pipeline(self, feature_cols):
        '''
        Pipeline for imputation and encoding
        '''
        model_config = self.config['model']['xgboost']
        handle_missing = self.config['preprocessing']['handle_missing']

        if handle_missing == 'native':
            numeric_step = ('features', 'passthrough', feature_cols)
        else:
            numeric_step = ('imputer', SimpleImputer(strategy=handle_missing), feature_cols)

        preprocessor = ColumnTransformer(
            transformers=[
                numeric_step,
                ('sex_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['Sex'])
            ], remainder='drop'
        )
        return Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(**model_config))
        ])

    def build_linear_pipeline(self, feature_cols):
        '''
        Linear baseline for the evaluation report (comparison only).
        LR can't take NaNs, so always impute, scale numerics, and one-hot encode for Sex.
        '''
        handle_missing = self.config['preprocessing']['handle_missing']
        impute_strategy = 'median' if handle_missing == 'native' else handle_missing

        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', Pipeline([
                    ('imputer', SimpleImputer(strategy=impute_strategy)),
                    ('scaler', StandardScaler())
                ]), feature_cols),
                ('sex_encoder', OneHotEncoder(handle_unknown='ignore'), ['Sex'])
            ], remainder='drop'
        )
        return Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
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
        test_df = df[df['Date'] >= test_date]
        self.train_df, self.val_df, self.test_df = train_df, val_df, test_df

        cols = feature_cols + ['Sex']

        val_pipeline = self.build_pipeline(feature_cols)
        val_pipeline.fit(train_df[cols], train_df['TotalKg'])
        val_pred = val_pipeline.predict(val_df[cols])
        self.print_metrics('VALIDATION', val_df['TotalKg'], val_pred)

        # out-of-sample residual spread -> prediction intervals at serving time
        residuals = val_df['TotalKg'].to_numpy() - val_pred
        self.val_pred = val_pred
        self.val_residuals = residuals
        self.residual_quantiles = self._build_intervals(residuals, val_df['days_since_last_meet'])

        pipeline = self.build_pipeline(feature_cols)
        pipeline.fit(pd.concat([train_df, val_df])[cols], pd.concat([train_df, val_df])['TotalKg'])
        test_pred = pipeline.predict(test_df[cols])
        self.pipeline = pipeline
        self.print_metrics('TEST', test_df['TotalKg'], test_pred)
        return pipeline

    @staticmethod
    def _build_intervals(residuals, days_since_last_meet, min_bucket_rows=100):
        '''
        Global q10/q90 band plus one band per layoff bucket (same bins as the evaluation report). 
        Residual spread grows with layoff length, so a single global band doesn't cover well with returining lifters. 
        Buckets with too few validation rows fall back to the global quantiles rather than serving noisy ones. 
        '''
        global_q10, global_q90 = np.quantile(residuals, [0.1, 0.9])
        buckets = pd.cut(days_since_last_meet, bins=Evaluator.GAP_BINS, labels=Evaluator.GAP_LABELS)

        by_layoff = []
        for label, upper in zip(Evaluator.GAP_LABELS, Evaluator.GAP_BINS[1:]):
            bucket_residuals = residuals[(buckets == label).to_numpy()]
            if len(bucket_residuals) >= min_bucket_rows:
                q10, q90 = np.quantile(bucket_residuals, [0.1, 0.9])
            else:
                q10, q90 = global_q10, global_q90
            by_layoff.append({
                'label': label,
                'max_days': None if np.isinf(upper) else upper, # None for json-safe >2 years bucket
                'q10': float(q10),
                'q90': float(q90),
                'n': int(len(bucket_residuals))
            })

        return {
            'global': {'q10': float(global_q10), 'q90': float(global_q90)},
            'by_layoff': by_layoff
        }

    def save_model(self, save_path='models/XGBR_model_v1.pkl'):
        joblib.dump(self.pipeline, save_path)

    def save_intervals(self, save_path='models/prediction_intervals.json'):
        '''
        residual quantiles from the validation fold (global + per layoff bucket),
        to show a prediction range instead of a bare point estimate for streamlit
        '''
        with open(save_path, 'w') as f:
            json.dump(self.residual_quantiles, f)
        print(f'Prediction intervals saved to "{save_path}"')

    def evaluation(self, report_path='reports/evaluation.md'):
        feature_cols = self.config['features']['columns']
        cols = feature_cols + ['Sex']

        # fit on the same train+val data as the final model so the comparison is fair
        fit_df = pd.concat([self.train_df, self.val_df])
        linear_pipeline = self.build_linear_pipeline(feature_cols)
        linear_pipeline.fit(fit_df[cols], fit_df['TotalKg'])

        evaluator = Evaluator(feature_cols=feature_cols)

        md = evaluator.report(self.pipeline, self.train_df, self.test_df, save_path=report_path,
                              comparison_pipelines={'Linear regression': linear_pipeline})
        evaluator.residual_plot(self.val_residuals, self.val_pred, self.val_df,
                                self.residual_quantiles['global'], save_path='reports/residual_plot.png')
        return md

if __name__ == '__main__':
    df = pd.read_csv('data/3-features/opl_features_IPF.csv')
    train = TrainingPipeline()
    train.train_from_data(df)
    train.save_model()
    train.save_intervals()
    train.evaluation()