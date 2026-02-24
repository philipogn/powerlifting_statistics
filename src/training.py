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
    def __init__(self, config_path: str='config/local.yaml',save_model: bool=False):
        self.config = yaml.safe_load(open(config_path))
        self.save_model = save_model

    def build_pipeline(self, feature_cols):
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

        if self.save_model:
            joblib.dump(pipeline, 'models/XGBR_model_v1.pkl')

        print(f'Mean Absolute Error: {mean_absolute_error(y_test, prediction):.4f}')
        print(f'Root Mean Squared Error: {root_mean_squared_error(y_test, prediction):.4f}')
        print(f'R2 Score: {r2_score(y_test, prediction):.4f}')
        return pipeline
        # SAVE MODEL AND USE PIPELINE METHODS ON API

if __name__ == '__main__':
    # raw data file
    df = pd.read_csv('data/3-features/opl_features_IPF.csv')
    train = TrainingPipeline(save_model=True)
    train.train_from_data(df)