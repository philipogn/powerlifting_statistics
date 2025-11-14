import pandas as pd

class DataCleaning():
    def __init__(self):
        pass
    
    '''
    Event: only SBD
    Equipment: Raw only
    Age: drop empty fields
    Division: Open, MR-O, FR-O only, as these are the standard
    TotalKg: drop empty fields (no DQs)
    Place: must be a number (no DQs)
    ParentFederation: only IPF
    Best3s: ensure all 3 lifts are successful (no DQs)
    '''

    def data_cleaning(df):
        print('Started data cleaning...')
        data = df[
            (df['Event'] == 'SBD') & 
            (df['Equipment'] == 'Raw') &
            (df['Age'].notna()) &
            (df['Division'].isin(['Open', 'MR-O', 'FR-O'])) &
            (df['TotalKg'].notna()) &
            (df['Place'].str.isnumeric()) &
            (df['ParentFederation'] == 'IPF') &
            (df['Best3SquatKg'].notna()) &
            (df['Best3BenchKg'].notna()) &
            (df['Best3DeadliftKg'].notna()) 
        ].copy()

        essential_columns = [
            'Name', 'Date', 'Sex', 'Age', 'BodyweightKg',
            'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 
            'TotalKg', 'Dots'
        ]

        data = data[essential_columns].copy()

        # Convert to datetime and sort by lifter and date (important for feature engineering)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(['Name', 'Date']).reset_index(drop=True)

        data.to_csv('data/2-preprocessed/cleanIPF_minimal.csv', index=False)
        print('Successfully cleaned data and saved to "data/2-preprocessed/cleanIPF_minimal.csv"')
        return data

if __name__ == '__main__':
    file_path = pd.read_csv('data/1-raw/openpowerlifting-2025-09-27-31932eca.csv')
    DataCleaning.data_cleaning(file_path)