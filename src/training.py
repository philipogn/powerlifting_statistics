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
    def __init__(self):
        pass

    def build_pipeline(self, config):
        model_config = config['model']['xgboost']
        feature_cols = config['features']['columns']

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
        config = yaml.safe_load(open('config/local.yaml'))

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Name', 'Date'])

        split_date = df['Date'].quantile(0.8)
        
        train_df, test_df = df[df["Date"] < split_date], df[df["Date"] >= split_date]
        y_train, y_test = train_df["TotalKg"], test_df["TotalKg"]

        pipeline = self.build_pipeline(config)
        pipeline.fit(train_df, y_train)
        prediction = pipeline.predict(test_df)

        print(f'Mean Absolute Error: {mean_absolute_error(y_test, prediction):.4f}')
        print(f'Root Mean Squared Error: {root_mean_squared_error(y_test, prediction):.4f}')
        print(f'R2 Score: {r2_score(y_test, prediction):.4f}')
        # SAVE MODEL AND USE PIPELINE METHODS ON API

if __name__ == '__main__':
    # raw data file
    df = pd.read_csv('data/3-features/openpowerlifting_features.csv')
    train = TrainingPipeline()
    train.train_from_data(df)