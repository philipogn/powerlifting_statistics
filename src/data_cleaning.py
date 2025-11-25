import pandas as pd

class DataCleaningConfig():
    '''
    Configuration constants for data cleaning
    And essential columns for model and feature engineer (dots could be useful, but ensure no data leakage)
    '''
    EVENT = 'SBD'
    EQUIPMENT = 'Raw'
    DIVISION = ['Open', 'MR-O', 'FR-O', 
                'Juniors', 'MR-Jr', 'FR-Jr', 'Sub-Juniors',
                'Masters 1', 'Masters 2', 'Masters 3', 'Masters 4', 'Masters 5'
                ]
    PARENT_FED = 'IPF'

    ESSENTIAL_COLUMNS = [
        'Name', 'Date', 'Sex', 'Age', 'BodyweightKg', 'Division',
        'Squat1Kg', 'Squat2Kg', 'Squat3Kg',
        'Bench1Kg', 'Bench2Kg', 'Bench3Kg', 
        'Deadlift1Kg', 'Deadlift2Kg', 'Deadlift3Kg', 
        'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 
        'TotalKg' #, 'Dots'
    ]


class DataProcessor(DataCleaningConfig):
    def __init__(self, raw_df: pd.DataFrame, save_path: str):
        self.df = raw_df
        self.save_path = save_path
        self.prioritise_divisions = [
            'Sub-Juniors', 'Juniors', 'MR-Jr', 'FR-Jr',
            'Masters 1', 'Masters 2', 'Masters 3', 'Masters 4', 'Masters 5'
            ]

    def select_target_data(self, df):
        '''
        Event: only SBD
        Equipment: Raw only
        Division: all common/important divisions, filtering out superfluous divs
        ParentFederation: only IPF
        '''
        print('Filtering to only target data (IPF, Raw, SBD)...')
        data = df[
            (df['Event'] == self.EVENT) & 
            (df['Equipment'] == self.EQUIPMENT) &
            (df['Division'].isin(self.DIVISION)) &
            (df['ParentFederation'] == self.PARENT_FED)
        ].copy()
        return data

    def remove_duplicate_entries(self, df):
        # checking columns of duplicate values, should just work with 'Name' and 'Date'
        duplicate_cols = ['Name', 'Date', 'TotalKg']

        df['is_junior'] = df['Division'].isin(self.prioritise_divisions).astype(int)
        df['priority'] = df['is_junior'].apply(lambda x: 1 if x == 1 else 2)

        df_clean = df.sort_values('priority').drop_duplicates(
            subset=duplicate_cols,
            keep='first'
        ).drop(columns=['is_junior', 'priority'])

        return df_clean


    def data_cleaning(self, df):
        '''
        TotalKg: drop empty fields (no disqualifications)
        Place: must be a number (no disqualifications)
        Best3s: ensure all 3 lifts are successful (no disqualifications)
        '''

        print('Cleaning and preprocessing data...')
        data = df[
            (df['TotalKg'].notna()) &
            (df['Place'].str.isnumeric()) &
            (df['Best3SquatKg'].notna()) &
            (df['Best3BenchKg'].notna()) &
            (df['Best3DeadliftKg'].notna()) 
        ].copy()

        data = data[self.ESSENTIAL_COLUMNS].copy()

        # Convert to datetime and sort by lifter and date (important for time-based feature engineering like calculating progression)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(['Name', 'Date']).reset_index(drop=True)
        return data

    def convert_to_csv(self, data):
        data.to_csv(self.save_path, index=False)
        print(f'Successfully cleaned data and saved to "{self.save_path}"')
        return data

    def run(self):
        target_data = self.select_target_data(self.df)
        clean_dupes = self.remove_duplicate_entries(target_data)
        clean_data = self.data_cleaning(clean_dupes)
        self.convert_to_csv(clean_data)

if __name__ == '__main__':
    raw_data = pd.read_csv('data/1-raw/openpowerlifting-2025-09-27.csv')
    # save_path = 'data/1-raw/openpowerlifting-2025-09-27-IPF-clean.csv'
    save_path = 'data/2-preprocessed/openpowerlifting-IPF-clean.csv'
    preprocess = DataProcessor(raw_data, save_path)
    preprocess.run()