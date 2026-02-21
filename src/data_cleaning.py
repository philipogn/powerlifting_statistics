import pandas as pd

class DataProcessor():
    ESSENTIAL_COLUMNS = [
        'Name', 'Date', 'Sex', 'Age', 'BodyweightKg',
        'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 
        'TotalKg', 'ParentFederation'
    ]

    def __init__(self, save_path: str=None, save_to_csv: bool=False, event: str='SBD', equipment: str='Raw'):
        self.save_path = save_path
        self.save_to_csv = save_to_csv
        self.event = event
        self.equipment = equipment

    def _select_target_data(self, df):
        '''
        SBD Event, Raw Equipment only
        Division: all common/important divisions, filtering out superfluous divs
        '''
        return df[
            (df['Sex'].isin(['M', 'F'])) & 
            (df['Event'] == self.event) & 
            (df['Equipment'] == self.equipment)
        ]

    def _remove_duplicate_entries(self, df):
        '''
        Removes possible duplicate entries, some lifters are eligible for 1+ Divisions
        Checks for duplicates based on repeating values of Name, Date, MeetName
        '''
        return df.drop_duplicates(subset=['Name', 'Date', 'MeetName'], keep='first')

    def _remove_invalid(self, df):
        '''
        Removes empty TotalKg, Place != int (no DQ's) and invalid lift attempts (weight < 20kg)
        Returns dataframe sorted by Name and Date, keeping only ESSENTIAL_COLUMNS
        '''
        data = df[
            (df['TotalKg'].notna()) &
            (df['Place'].str.isnumeric()) &
            (df['Best3SquatKg'] >= 20) &
            (df['Best3BenchKg'] >= 20) &
            (df['Best3DeadliftKg'] >= 20)
        ].copy()
        data = data[self.ESSENTIAL_COLUMNS].copy()

        return data

    def _flag_anomaly(self, df):
        '''
        Flagging outliers/anomaly lifts, weirdly proportion ratios of SBD
        Could be due to injury prior to competition, lifter attempting max weight possible to not get DQ'ed
        '''
        squat_anomaly = df['Best3SquatKg'] < (0.5 * df[['Best3BenchKg','Best3DeadliftKg']].mean(axis=1))
        bench_anomaly = df['Best3BenchKg'] < (0.3 * df[['Best3SquatKg','Best3DeadliftKg']].mean(axis=1))
        deadlift_anomaly = df['Best3DeadliftKg'] < (0.8 * df[['Best3SquatKg','Best3BenchKg']].mean(axis=1))
        df['anomaly'] = squat_anomaly | bench_anomaly | deadlift_anomaly
        return df

    def _convert_to_csv(self, data):
        data.to_csv(self.save_path, index=False)
        print(f'Successfully cleaned data and saved to "{self.save_path}"')
        return data

    def transform(self, df):
        df = df.copy()
        df = self._select_target_data(df)
        df = self._remove_duplicate_entries(df)
        df = self._remove_invalid(df)
        df = self._flag_anomaly(df)

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Name', 'Date']).reset_index(drop=True)

        if self.save_to_csv:
            self._convert_to_csv(df)
        return df

if __name__ == '__main__':
    raw_data = pd.read_csv('data/1-raw/openpowerlifting-2025-09-27.csv')
    save_path = 'data/2-preprocessed/openpowerlifting_preprocessed.csv'
    preprocess = DataProcessor(save_path, save_to_csv=True)
    preprocess.transform(raw_data)