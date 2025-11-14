import pandas as pd

class FeatureEngineering():
    def __init__(self, dataframe, min_meets=2):
        self.dataframe = dataframe
        self.min_meets = min_meets
        self.df_with_features = None
        
    def create_features(self, current_meet, previous_meet):
        features = current_meet.copy()

        # 1. previous performance
        features['prev_squat'] = previous_meet['Best3SquatKg'].iloc[-1]
        features['prev_bench'] = previous_meet['Best3BenchKg'].iloc[-1]
        features['prev_deadlift'] = previous_meet['Best3DeadliftKg'].iloc[-1]
        features['prev_total'] = previous_meet['TotalKg'].iloc[-1]

        # 2. historical performances, averaged, min, max
        features['avg_total'] = previous_meet['TotalKg'].mean()
        features['max_total_ever'] = previous_meet['TotalKg'].max()
        features['min_total_ever'] = previous_meet['TotalKg'].min()

        features['avg_squat'] = previous_meet['Best3SquatKg'].mean()
        features['avg_bench'] = previous_meet['Best3BenchKg'].mean()
        features['avg_deadlift'] = previous_meet['Best3DeadliftKg'].mean()

        # 3. improvement rates over meets, 
        if len(previous_meet) > 1:
            first_total = previous_meet['TotalKg'].iloc[0]
            last_total = previous_meet['TotalKg'].iloc[-1]
            num_meets = len(previous_meet) - 1 # n meets -> n - 1 intervals

            features['total_gain_per_meet'] = (last_total - first_total) / num_meets

            features['squat_gain_per_meet'] = (
                previous_meet['Best3SquatKg'].iloc[-1] - previous_meet['Best3SquatKg'].iloc[0]) / num_meets
            features['bench_gain_per_meet'] = (
                previous_meet['Best3BenchKg'].iloc[-1] - previous_meet['Best3BenchKg'].iloc[0]) / num_meets
            features['deadlift_gain_per_meet'] = (
                previous_meet['Best3DeadliftKg'].iloc[-1] - previous_meet['Best3DeadliftKg'].iloc[0]) / num_meets
        else:
            # else all 0
            features['total_gain_per_meet'] = 0
            features['squat_gain_per_meet'] = 0
            features['bench_gain_per_meet'] = 0
            features['deadlift_gain_per_meet'] = 0
        
        return features

    def process_single_lifter(self, lifter_data):
        lifting_data = []

        for i in range(1, len(lifter_data)):
            current = lifter_data.iloc[i]
            previous = lifter_data.iloc[:i]
            features = self.create_features(current, previous)
            lifting_data.append(features)
        return pd.DataFrame(lifting_data)

    # issue with <x>_gain_per_meet resulting iin 0 value
    # could change to requires 3 meets?
    def engineer_features(self):
        df = self.dataframe.sort_values(['Name', 'Date']).reset_index(drop=True) # sort again for now, might need to modify preprocessing later
        all_lifting_data = []
        skipped_count = 0

        print(f"Started feature engineering...")
        for lifter_name, lifter_data in df.groupby('Name'): # for each lifters competition history
            if len(lifter_data) < self.min_meets: # only can predict lifters with at least two comp history
                skipped_count += 1
                continue
            features = self.process_single_lifter(lifter_data)
            all_lifting_data.append(features)

        self.df_with_features = pd.concat(all_lifting_data)

        print(f"\nFeature engineering complete!")
        print(f"Training examples created: {len(self.df_with_features)}")
        print(f"Skipped {skipped_count} lifters with <{self.min_meets} meets")
        print(f"{self.df_with_features['Name'].nunique()} lifters with {self.min_meets}+ meets")
        return self.df_with_features
    
    def save_features(self, output_path='data/3-features/engineered_features.csv'):
        if self.df_with_features is None:
            raise ValueError('\nNo features to save, run engineer_features() first')
        self.df_with_features.to_csv(output_path, index=False)
        print(f"\nFeatures saved to {output_path}")

    def run(self, output_path='data/3-features/engineered_features.csv'):
        self.engineer_features()
        self.save_features(output_path)
        return self.df_with_features

    


if __name__ == '__main__':
    df = pd.read_csv('data/2-preprocessed/cleanIPF_minimal.csv')
    features = FeatureEngineering(df)
    features.run()

