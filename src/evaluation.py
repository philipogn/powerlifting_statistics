from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import os


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
            "Persistance (previous total)": self._metrics(y, test_df["prev_total"])
        }

    def _segment_table(self, test_df, model_pred, column, bins, labels):
        seg = test_df[[column, self.target, "prev_total"]].copy()
        seg["model_err"] = np.abs(seg[self.target] - model_pred)
        seg["persistance_err"] = np.abs(seg[self.target] - seg["prev_total"])
        seg["bucket"] = pd.cut(seg[column], bins=bins, labels=labels)
        seg = pd.DataFrame(seg)
        return (
            seg.groupby("bucket", observed=True)
                .agg(n=("model_err", "size"), model_mae=("model_err", "mean"), persistance_mae=("persistance_err", "mean"))
                .round(1)
        )

    def evaluate(self, pipeline, train_df, test_df):
        X_test = test_df[self.feature_cols + ["Sex"]]
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

    def to_markdown(self, results, segments, train_df, test_df):
        persist_mae = results["Persistance (previous total)"]["MAE"]
        model_mae = results["Model"]["MAE"]
        improvement = (1 - (model_mae / persist_mae)) * 100

        lines = [
            f"# Model Evaluation\n",

            f"Time-base split.\n"
            f"Train: {train_df["Date"].min().date()} to {train_df["Date"].max().date()}",
            f"Test: {test_df["Date"].min().date()} to {test_df["Date"].max().date()}\n\n",

            f"Model MAE is **{model_mae:.1f} kg** vs **{persist_mae:.1f} kg** for simply repeating the lifter's previous total, "
            f"a **{improvement:.1f}% reduction in error** over the persistance baseline\n",

            f"## Model vs baselines",
            f"{pd.DataFrame(results).T.round(3).to_markdown()}",

            f"## Error by segment (MAE, kg)"
        ]
        for title, table in segments.items():
            lines += [f"### {title}", "", f"{table.to_markdown()}", ""]
        return "\n".join(lines)

    def report(self, pipeline, train_df, test_df, save_path=None):
        results, segments = self.evaluate(pipeline, train_df, test_df)
        md = self.to_markdown(results, segments, train_df, test_df)
        if save_path:
            report_dir = os.path.dirname(save_path)
            if report_dir:
                os.makedirs(report_dir, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(md)
            print(f"Evaluation saved to '{save_path}'")
        return md
