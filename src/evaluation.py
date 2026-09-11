from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import numpy as np
# predict the training mean baseline
# persistance baseline (predict prev total)

class Evaluator():
    GAP_BINS = [0, 90, 180, 365, 730, np.inf] # days
    GAP_LABELS = ["<3 months", "3-6 months", "6-12 months", "1-2 years", ">2 years"]
    MEET_BINS = [0, 3, 5, 10, np.inf]
    MEET_LABELS = ["2-3", "4-5", "6-10", ">10"]

    def __init__(self, feature_cols, target="TotalKg"):
        self.feature_cols = feature_cols
        self.target = target

    @staticmethod
    def _metrics(y_true, y_pred):
        res = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }
        return res

    def _baselines(self, train_df, test_df):
        y = test_df[self.target]
        base = {
            "Predict training mean": self._metrics(y, np.full(len(y), train_df[self.target].mean())),
            "Persistance (previous total)": self._metrics(y, test_df['previous_total'])
        }

    def evaluate(self, pipeline, train_df, test_df):
        X_test = test_df[self.feature_cols]
        prediction = pipeline.predict(X_test)

        results = self._baselines(train_df, test_df)
        results["Model"] = self._metrics(test_df[self.target], prediction)

        return results

    def to_markdown(self):
        pass


