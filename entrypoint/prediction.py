import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def predict_unseen(model_path, dataset):
    data = pd.read_csv(dataset)
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    prediction = model.predict(data)
    actual = data['TotalKg']

    print(f'Mean Absolute Error: {mean_absolute_error(actual, prediction):.4f}')
    print(f'Root Mean Squared Error: {root_mean_squared_error(actual, prediction):.4f}')
    print(f'R2 Score: {r2_score(actual, prediction):.4f}')
    return prediction


model = 'models/XGBR_model_v1.pkl'
unseen_data = 'data/3-features/opl_features_not_IPF.csv'
predict_unseen(model_path=model, dataset=unseen_data)