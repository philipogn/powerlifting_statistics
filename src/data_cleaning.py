import pandas as pd

class DataProcessor():
    EVENT = 'SBD'
    EQUIPMENT = 'Raw'
    ESSENTIAL_COLUMNS = [
        'Name', 'Date', 'Sex', 'Age', 'BodyweightKg',
        'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 
        'TotalKg', 'ParentFederation'
    ]

    def __init__(self, raw_df: pd.DataFrame, save_path: str, save_to_csv: bool=False):
        self.df = raw_df
        self.save_path = save_path
        self.save_to_csv = save_to_csv

    def _select_target_data(self, df):
        '''
        SBD Event, Raw Equipment only
        Division: all common/important divisions, filtering out superfluous divs
        '''
        return df[
            (df['Sex'].isin(['M', 'F'])) & 
            (df['Event'] == self.EVENT) & 
            (df['Equipment'] == self.EQUIPMENT)
        ]

    def _remove_duplicate_entries(self, df):
        '''
        Removes possible duplicate entries, some lifters are eligible for 1+ Divisions
        Checks for duplicates based on repeating values of Name, Date, MeetName
        '''
        return df.drop_duplicates(subset=['Name', 'Date', 'MeetName'], keep='first')

    def _remove_invalid(self, df):
        '''
        Removes empty TotalKg, and where Place != int (no disqualifications)
        Removes invalid lift attempts (min weight = 20kg)

        Returns dataframe sorted by Name and Date, keeping only ESSENTIAL_COLUMNS
        '''
        print('Cleaning and preprocessing data...')
        data = df[
            (df['TotalKg'].notna()) &
            (df['Place'].str.isnumeric()) &
            (df['Best3SquatKg'].notna()) & (df['Best3SquatKg'] >= 20) &
            (df['Best3BenchKg'].notna()) & (df['Best3BenchKg'] >= 20) &
            (df['Best3DeadliftKg'].notna()) & (df['Best3DeadliftKg'] >= 20)
        ].copy()
        data = data[self.ESSENTIAL_COLUMNS].copy()

        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(['Name', 'Date']).reset_index(drop=True)
        return data

    def _flag_anomaly(self, df):
        # need some function to ensure valid data/lifts or remove outliers?
        # data = data[data['TotalKg'] >= 200]
        # new method, to mark weirdly proportioned lifts as anomaly/outliers
        # filtered from EDA discovery
        df['squat_anomaly'] = df['Best3SquatKg'] < (0.5 * df[['Best3BenchKg','Best3DeadliftKg']].mean(axis=1))
        df['bench_anomaly'] = df['Best3BenchKg'] < (0.3 * df[['Best3SquatKg','Best3DeadliftKg']].mean(axis=1))
        df['deadlift_anomaly'] = df['Best3DeadliftKg'] < (0.8 * df[['Best3SquatKg','Best3BenchKg']].mean(axis=1))
        df['anomaly'] = df[['squat_anomaly', 'bench_anomaly', 'deadlift_anomaly']].any(axis=1)
        df = df.drop(labels=['squat_anomaly', 'bench_anomaly', 'deadlift_anomaly'], axis='columns')
        return df

    def _convert_to_csv(self, data):
        data.to_csv(self.save_path, index=False)
        print(f'Successfully cleaned data and saved to "{self.save_path}"')
        return data

    def run_data_cleaner(self):
        target_data = self._select_target_data(self.df)
        clean_dupes = self._remove_duplicate_entries(target_data)
        clean_invalid = self._remove_invalid(clean_dupes)
        clean_data = self._flag_anomaly(clean_invalid)
        if self.save_to_csv:
            self._convert_to_csv(clean_data)
        return clean_data

if __name__ == '__main__':
    raw_data = pd.read_csv('data/1-raw/openpowerlifting-2025-09-27.csv')
    save_path = 'data/2-preprocessed/openpowerlifting_preprocessed.csv'
    preprocess = DataProcessor(raw_data, save_path, save_to_csv=True)
    preprocess.run_data_cleaner()