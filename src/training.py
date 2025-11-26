import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()])

logger = logging.getLogger(__name__)


class TrainingPipeline():
    def __init__(self, dataframe, config):
        self.dataframe = dataframe
        self.config = config
        self.feature_columns = config["features"]["columns"]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_df = None
        self.test_df = None
        self.predictions = None
        self.model = None


    def prepare_dataset(self):
        logger.info('Encoding and splitting dataset...')
        scaler = StandardScaler()
        self.dataframe['Sex'] = self.dataframe['Sex'].astype('category') # change type for xgboost
        self.dataframe[self.feature_columns] = self.dataframe[self.feature_columns].fillna(0)

        # Time-base train-test split
        self.dataframe['Date'] = pd.to_datetime(self.dataframe['Date'])
        self.dataframe = self.dataframe.sort_values('Date').reset_index(drop=True)

        split_date = self.dataframe['Date'].quantile(0.8)
        self.train_df = self.dataframe[self.dataframe['Date'] < split_date]
        self.test_df = self.dataframe[self.dataframe['Date'] >= split_date]

        logger.info(f"""
                    ==Data split==
                    Training date range: {self.train_df['Date'].min()} to {self.train_df['Date'].max()}
                    Testing date range: {self.test_df['Date'].min()} to {self.test_df['Date'].max()}
                    Training examples: {len(self.train_df)}
                    Testing examples: {len(self.test_df)}""")

        self.X_train = self.train_df[self.feature_columns]
        self.X_test = self.test_df[self.feature_columns]
        self.y_train = self.train_df['TotalKg']
        self.y_test = self.test_df['TotalKg']

    def train_models(self):
        logger.info('Training Gradient Boosting Regressor model')
        xgb_cfg = self.config['model']['xgboost']
        model = XGBRegressor(**xgb_cfg)
        model.fit(self.X_train, self.y_train)
        model_y_pred = model.predict(self.X_test)
        self.predictions = model_y_pred
        self.model = model

    # Evaluate models
    def evaluate_models(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        logger.info(f"""
                    Mean Absolute Error: {mae:.4f}
                    Mean Squared Error: {mse:.4f}
                    Root Mean Squared Error: {rmse:.4f}
                    R2 Score: {r2:.4f}""")

        return f'MAE: {mae}, MSE: {mse}, RSME: {rmse}, R2: {r2}'

    def save_model(self):
        path = config['data']['model_output_path']
        joblib.dump(self.model, path)

    def run(self):
        self.prepare_dataset()
        self.train_models()
        result = self.evaluate_models(self.y_test, self.predictions)

        print(result)
        # result.to_csv('data/4-predictions/model_evaluation.csv')
        self.save_model()
    

if __name__ == '__main__':
    config = yaml.safe_load(open("config/local.yaml"))
    df = pd.read_csv(config['data']['train_path'])
    train = TrainingPipeline(df, config)
    train.run()