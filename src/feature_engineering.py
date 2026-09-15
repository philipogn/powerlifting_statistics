import pandas as pd
from tqdm import tqdm

OUTPUT_COLS = ['Name', 'Date', 'Sex', 'Age', 'BodyweightKg', 'TotalKg', 
               'prev_squat', 'prev_bench', 'prev_deadlift', 'prev_total',
               'avg_squat', 'avg_bench', 'avg_deadlift', 
               'days_since_last_meet', 'total_meets', 
               'percent_gain_since_last', 'career_avg_improvement_rate', 'total_std']

class FeatureEngineering():
    def __init__(self, save_path: str=None, save_to_csv: bool=False):
        self.save_path = save_path
        self.save_to_csv = save_to_csv
        self.min_meets = 3

    def create_features(self, current_meet, previous_meet):
        '''
        Creates features base on previous SBD performance and average, 
        time-based features, percentage gain and improvement

        Shared by training (per historical meet) and serving (src/inference_service.py), so the two can never drift apart.
        current_meet only needs a 'Date'. previous_meet needs Date, TotalKg and the Best3*Kg columns, sorted chronologically.
        '''
        features = {}

        features['prev_squat'] = previous_meet['Best3SquatKg'].iloc[-1]
        features['prev_bench'] = previous_meet['Best3BenchKg'].iloc[-1]
        features['prev_deadlift'] = previous_meet['Best3DeadliftKg'].iloc[-1]
        features['prev_total'] = previous_meet['TotalKg'].iloc[-1]  # persistence baseline

        features['avg_squat'] = previous_meet['Best3SquatKg'].mean()
        features['avg_bench'] = previous_meet['Best3BenchKg'].mean()
        features['avg_deadlift'] = previous_meet['Best3DeadliftKg'].mean()

        features['days_since_last_meet'] = (
            pd.to_datetime(current_meet['Date']) - pd.to_datetime(previous_meet['Date'].iloc[-1])
        ).days
        features['total_meets'] = len(previous_meet)

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
        '''
        Creates features based on grouped lifter data, 
        a row is emitted only when lifter has at least min_meets - 1 earlier meets, 
        so the filter depends purely on each rows past and never on the lifters future meets
        Returns list of dictionaries with features for each qualifying meet
        '''
        lifting_data = []
        for i in range(self.min_meets - 1, len(lifter_data)):
            current = lifter_data.iloc[i]
            previous = lifter_data.iloc[:i]
            meet = current.to_dict()
            meet.update(self.create_features(current, previous))
            lifting_data.append(meet)
        return lifting_data

    def _save_features(self, df):
        df.to_csv(self.save_path, index=False)
        return df

    def engineer_features(self, df):
        '''
        Sorts by name and date, then groups by name and creates features for each lifter
        Lifters with fewer than min_meets total meets contribute no rows (loop bound in _process_lifter),
        to prevent unstable features without conditioning on future meet counts
        '''
        df = df.sort_values(['Name', 'Date']).reset_index(drop=True)
        all_lifting_data = []
        for name, lifter_data in tqdm(df.groupby('Name'), desc='Engineering Features...'):
            all_lifting_data.extend(self._process_lifter(lifter_data))
        
        result = pd.DataFrame(all_lifting_data)[OUTPUT_COLS].round(5)
        if self.save_to_csv:
            self._save_features(result)
        
        return result


if __name__ == '__main__':
    df = pd.read_csv('data/2-preprocessed/opl_preprocessed_IPF.csv')
    save_path = 'data/3-features/opl_features_IPF.csv'

    features = FeatureEngineering(save_path=save_path, save_to_csv=True)
    features.engineer_features(df)

