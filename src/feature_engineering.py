import pandas as pd
import yaml
from tqdm import tqdm

OUTPUT_COLS = ['Name', 'Date', 'Sex', 'Age', 'BodyweightKg', 'TotalKg', 
               'prev_squat', 'prev_bench', 'prev_deadlift', 
               'avg_squat', 'avg_bench', 'avg_deadlift', 
               'days_since_last_meet', 'total_meets', 
               'percent_gain_since_last', 'career_avg_improvement_rate', 'total_std']

class FeatureEngineering():
    def __init__(self, save_path: str=None, save_to_csv: bool=False, min_meets=3):
        self.save_path = save_path
        self.save_to_csv = save_to_csv
        self.min_meets = min_meets

    def _create_features(self, current_meet, previous_meet):
        features = {}

        features['prev_squat'] = previous_meet['Best3SquatKg'].iloc[-1]
        features['prev_bench'] = previous_meet['Best3BenchKg'].iloc[-1]
        features['prev_deadlift'] = previous_meet['Best3DeadliftKg'].iloc[-1]

        features['avg_squat'] = previous_meet['Best3SquatKg'].mean()
        features['avg_bench'] = previous_meet['Best3BenchKg'].mean()
        features['avg_deadlift'] = previous_meet['Best3DeadliftKg'].mean()

        features['days_since_last_meet'] = (
            pd.to_datetime(current_meet['Date']) - pd.to_datetime(previous_meet['Date'].iloc[-1])
        ).days
        features['total_meets'] = len(previous_meet)
        
        # total kg lifted to bodyweight ratio on previous meet
        features['total_bodyweight_ratio'] = previous_meet['TotalKg'].iloc[-1] / previous_meet['BodyweightKg'].iloc[-1]
        
        if len(previous_meet) >= 2:
            first, last, second_last = previous_meet['TotalKg'].iloc[0], previous_meet['TotalKg'].iloc[-1], previous_meet['TotalKg'].iloc[-2]
            features['percent_gain_since_last'] = ((last - second_last) / second_last)
            features['career_avg_improvement_rate'] = ((last - first) / first / (len(previous_meet) - 1))
        else:
            features['percent_gain_since_last'] = 0
            features['career_avg_improvement_rate'] = 0
        
        features['total_std'] = previous_meet['TotalKg'].std() if len(previous_meet) > 1 else 0
        
        return features

    def _process_lifter(self, lifter_data):
        lifting_data = []
        for i in range(1, len(lifter_data)):
            current = lifter_data.iloc[i]
            previous = lifter_data.iloc[:i]
            meet = current.to_dict()
            meet.update(self._create_features(current, previous))
            lifting_data.append(meet)
        # drop first meet, not useful as it returns null/0 on some features
        return lifting_data[1:] if len(lifting_data) > 1 else lifting_data

    def _save_features(self, df):
        df.to_csv(self.save_path, index=False)
        return df

    def engineer_features(self, df):
        df = df.sort_values(['Name', 'Date']).reset_index(drop=True) # sort by name, date
        all_lifting_data = []
        for name, lifter_data in tqdm(df.groupby('Name'), desc='Engineering Features...'):
            if len(lifter_data) < self.min_meets: # only can predict lifters with at least two comp history
                continue
            all_lifting_data.extend(self._process_lifter(lifter_data))
        
        result = pd.DataFrame(all_lifting_data)[OUTPUT_COLS].round(5)
        if self.save_to_csv:
            self._save_features(result)
        
        return result


if __name__ == '__main__':
    # config = yaml.safe_load(open('config/local.yaml'))
    # recheck df csv
    # df = pd.read_csv(config['data']['feature_engineer'])
    df = pd.read_csv('data/2-preprocessed/opl_preprocessed_IPF.csv')
    save_path = 'data/3-features/opl_features_IPF.csv'

    features = FeatureEngineering(save_path=save_path, save_to_csv=True)
    features.engineer_features(df)

