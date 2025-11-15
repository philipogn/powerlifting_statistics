import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()])

logger = logging.getLogger(__name__)


class TrainingPipeline():
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.feature_columns = [
            'prev_squat', 'prev_bench', 'prev_deadlift', 'prev_total',
            'avg_squat', 'avg_bench', 'avg_deadlift', 'avg_total',
            'max_total_ever', 'min_total_ever', 
            'squat_gain_per_meet', 'bench_gain_per_meet', 'deadlift_gain_per_meet', 'total_gain_per_meet',
            'Age', 'BodyweightKg'
        ]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_df = None
        self.test_df = None
        self.models = {}
        self.predictions = {}
        self.results = {}


    def prepare_dataset(self):
        logger.info('Encoding and splitting dataset...')
        self.dataframe['SexEncoded'] = self.dataframe['Sex'].map({'M': 1, 'F': 0})
        self.feature_columns.append('SexEncoded')
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
        logger.info('Training Logistic Regression model')
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        lr_y_pred = lr.predict(self.X_test)
        self.models['LR'] = lr
        self.predictions['LR'] = lr_y_pred

        logger.info('Training Random Forest Regressor model')
        rdr = RandomForestRegressor()
        rdr.fit(self.X_train, self.y_train)
        rdr_y_pred = rdr.predict(self.X_test)
        self.models['RFR'] = rdr
        self.predictions['RFR'] = rdr_y_pred

        logger.info('Training Gradient Boosting Regressor model')
        gbr = GradientBoostingRegressor()
        gbr.fit(self.X_train, self.y_train)
        gbr_y_pred = gbr.predict(self.X_test)
        self.models['GBR'] = gbr
        self.predictions['GBR'] = gbr_y_pred


    # Evaluate models
    def evaluate_models(self, y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        logger.info(f"""
                    ==={model_name}===
                    Mean Absolute Error: {mae:.4f}
                    Mean Squared Error: {mse:.4f}
                    Root Mean Squared Error: {rmse:.4f}
                    R2 Score: {r2:.4f}""")

        return {'MAE': mae, 'MSE': mse, 'RSME': rmse, 'R2': r2}

    def save_model(self):
        for name, model in self.models.items():
            joblib.dump(model, f'{name}_model.pkl')

    def run(self):
        self.prepare_dataset()
        self.train_models()
        for model_name, y_pred in self.predictions.items():
            self.results[model_name] = self.evaluate_models(self.y_test, y_pred, model_name)

        print('\n==Comparison==')
        comparison = pd.DataFrame(self.results).T
        print(comparison.round(2))

        comparison.to_csv('data/4-predictions/model_evaluation.csv')
        self.save_model()
    

if __name__ == '__main__':
    df = pd.read_csv('data/3-features/engineered_features.csv')
    train = TrainingPipeline(df)
    train.run()