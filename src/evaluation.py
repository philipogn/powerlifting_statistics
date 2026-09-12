from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd


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
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }


    def _baselines(self, train_df, test_df):
        """ The training mean baseline, and persistance baseline (prev total) """
        y = test_df[self.target]
        return {
            "Predict training mean": self._metrics(y, np.full(len(y), train_df[self.target].mean())),
            "Persistance (previous total)": self._metrics(y, test_df["previous_total"])
        }


    def _segment_table(self, test_df, model_pred, column, bins, labels):
        seg = test_df[[column, self.target, "prev_total"]].copy()
        seg["model_err"] = np.abs(seg[self.target] - model_pred)
        seg["persistance_err"] = np.abs(seg[self.target] - seg["prev_total"])
        seg["bucket"] = pd.cut(seg[column], bins=bins, labels=labels)
        seg = pd.DataFrame(seg)
        return (
            seg.groupby("bucket")
                .agg(n=("model_err", "size"), model_mae=("model_err", "mean"), persistance_mae=("persistance_err", "mean"))
                .round(1)
        )


    def evaluate(self, pipeline, train_df, test_df):
        X_test = test_df[self.feature_cols]
        prediction = pipeline.predict(X_test)

        results = self._baselines(train_df, test_df)
        results["Model"] = self._metrics(test_df[self.target], prediction)

        segments = {
            "Days since last meet": self._segment_table(
                test_df, prediction, "days_since_last_meet", self.GAP_BINS, self.GAP_LABELS),
            "Number of prior meets": self._segment_table(
                test_df, prediction, "total_meets", self.MEET_BINS, self.MEET_LABELS)
        }
        return results, segments


    def to_markdown(self):
        pass

    # def report(self, pipeline, train_df, test_df, save_path=None):
    #     results, segments = self.evaluate(pipeline, train_df, test_df)


