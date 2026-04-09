import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import joblib
import yaml

class TrainingPipeline():
    def __init__(self, config_path: str='config/local.yaml'):
        self.config = yaml.safe_load(open(config_path))
        self.pipeline = None

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

    def train_from_data(self, df):
        feature_cols = self.config['features']['columns']

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Name', 'Date'])

        split_date = df['Date'].quantile(0.8)
        
        train_df, test_df = df[df["Date"] < split_date], df[df["Date"] >= split_date]
        X_train, X_test = train_df[feature_cols + ['Sex']], test_df[feature_cols + ['Sex']]
        y_train, y_test = train_df["TotalKg"], test_df["TotalKg"]

        pipeline = self.build_pipeline(feature_cols)
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)

        self.pipeline = pipeline
        print(f'========= TRAINING RESULTS =========')
        print(f'Mean Absolute Error: {mean_absolute_error(y_test, prediction):.4f}')
        print(f'Root Mean Squared Error: {root_mean_squared_error(y_test, prediction):.4f}')
        print(f'R2 Score: {r2_score(y_test, prediction):.4f}')
        print(f'====================================')
        return pipeline
        # SAVE MODEL AND USE PIPELINE METHODS ON API

    def save_model(self, save_path='models/XGBR_model_v1.pkl'):
        joblib.dump(self.pipeline, save_path)

if __name__ == '__main__':
    df = pd.read_csv('data/3-features/opl_features_IPF.csv')
    train = TrainingPipeline(save_model=False)
    train.train_from_data(df)