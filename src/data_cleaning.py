import pandas as pd

class DataCleaningConfig():
    '''
    Configuration constants for data cleaning
    And ESSENTIAL_COLUMNS to keep for model and feature engineer
    '''
    EVENT = 'SBD'
    EQUIPMENT = 'Raw'
    PARENT_FED = 'IPF'
    ESSENTIAL_COLUMNS = [
        'Name', 'Date', 'Sex', 'Age', 'BodyweightKg',
        'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 
        'TotalKg', 'ParentFederation'
    ]


class DataProcessor(DataCleaningConfig):
    def __init__(self, raw_df: pd.DataFrame, save_path: str, save_to_csv: bool=False):
        self.df = raw_df
        self.save_path = save_path
        self.save_to_csv = save_to_csv

    def _select_target_data(self, df):
        '''
        SBD Event, Raw Equipment only
        Division: all common/important divisions, filtering out superfluous divs
        '''
        print('Filtering to only target data (Raw, SBD)...')
        data = df[
            (df['Event'] == self.EVENT) & 
            (df['Equipment'] == self.EQUIPMENT)
        ].copy()
        return data

    def _remove_duplicate_entries(self, df):
        '''
        Removes possible duplicate meet entries, as some lifters are eligible for 1+ Divisions
        Checks for duplicates based on repeating values of Name, Date, TotalKg
        '''
        duplicate_cols = ['Name', 'Date', 'TotalKg']

        df_clean = df.drop_duplicates(
            subset=duplicate_cols,
            keep='first'
        )

        return df_clean


    def _data_cleaning(self, df):
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

        # need some function to ensure valid data/lifts or remove outliers?
        # data = data[data['TotalKg'] >= 200]
        # not here

        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(['Name', 'Date']).reset_index(drop=True)
        return data

    def _convert_to_csv(self, data):
        data.to_csv(self.save_path, index=False)
        print(f'Successfully cleaned data and saved to "{self.save_path}"')
        return data

    def run_data_cleaner(self):
        target_data = self._select_target_data(self.df)
        clean_dupes = self._remove_duplicate_entries(target_data)
        clean_data = self._data_cleaning(clean_dupes)
        if self.save_to_csv:
            self._convert_to_csv(clean_data)

if __name__ == '__main__':
    raw_data = pd.read_csv('data/1-raw/openpowerlifting-2025-09-27.csv')
    save_path = 'data/2-preprocessed/openpowerlifting_preprocessed.csv'
    preprocess = DataProcessor(raw_data, save_path, save_to_csv=True)
    preprocess.run_data_cleaner()