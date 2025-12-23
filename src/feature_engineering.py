import pandas as pd
import yaml

class FeatureEngineering():
    def __init__(self, dataframe, min_meets=3):
        self.dataframe = dataframe
        self.min_meets = min_meets
        self.df_with_features = None
        
    def create_features(self, current_meet, previous_meet):
        features = current_meet.copy()

        # 1. previous performance
        features['prev_squat'] = previous_meet['Best3SquatKg'].iloc[-1]
        features['prev_bench'] = previous_meet['Best3BenchKg'].iloc[-1]
        features['prev_deadlift'] = previous_meet['Best3DeadliftKg'].iloc[-1]

        # 2. averaged of lifts
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
            features['percent_gain_since_last'] = (
                (previous_meet['TotalKg'].iloc[-1] - previous_meet['TotalKg'].iloc[-2]) / 
                previous_meet['TotalKg'].iloc[-2]
            )
            features['career_avg_improvement_rate'] = (
                (previous_meet['TotalKg'].iloc[-1] - previous_meet['TotalKg'].iloc[0]) / 
                previous_meet['TotalKg'].iloc[0] / (len(previous_meet) - 1)
            )
        else:
            features['percent_gain_since_last'] = 0
            features['career_avg_improvement_rate'] = 0
        
        features['total_std'] = previous_meet['TotalKg'].std() if len(previous_meet) > 1 else 0
        
        return features

    def process_single_lifter(self, lifter_data):
        lifting_data = []

        for i in range(1, len(lifter_data)):
            current = lifter_data.iloc[i]
            previous = lifter_data.iloc[:i]
            features = self.create_features(current, previous)
            lifting_data.append(features)
        return pd.DataFrame(lifting_data)

    def engineer_features(self):
        df = self.dataframe.sort_values(['Name', 'Date']).reset_index(drop=True) # sort by name, date
        all_lifting_data = []
        skipped_count = 0

        print(f"Started feature engineering...")
        for lifter_name, lifter_data in df.groupby('Name'): # for each lifters competition history
            if len(lifter_data) < self.min_meets: # only can predict lifters with at least two comp history
                skipped_count += 1
                continue
            if len(lifter_data) > self.min_meets:
                features = features.iloc[1:] # ignore first meet, not useful as it returns null/0 on some features
            features = self.process_single_lifter(lifter_data)
            all_lifting_data.append(features)

        self.df_with_features = pd.concat(all_lifting_data)

        print(f"\nFeature engineering complete!")
        print(f"Training examples created: {len(self.df_with_features)}")
        print(f"Skipped {skipped_count} lifters with <{self.min_meets} meets")
        print(f"{self.df_with_features['Name'].nunique()} lifters with {self.min_meets}+ meets")
        return self.df_with_features
    
    def save_features(self, dataset_type):
        if self.df_with_features is None:
            raise ValueError('\nNo features to save, run engineer_features() first')
        output_path = f'data/3-features/{dataset_type}_dataset.csv'
        self.df_with_features.to_csv(output_path, index=False)
    
    def run(self, dataset_type):
        self.engineer_features()
        self.save_features(dataset_type)
        return self.df_with_features


if __name__ == '__main__':
    config = yaml.safe_load(open('config/local.yaml'))
    df = pd.read_csv(config['data']['feature_engineer'])
    dataset_type = 'Train'

    features = FeatureEngineering(df)
    features.run(dataset_type=dataset_type)

