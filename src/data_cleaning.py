import pandas as pd

class DataCleaningConfig():
    # config constants for data cleaning
    EVENT = 'SBD'
    EQUIPMENT = 'Raw'
    DIVISION = ['Open', 'MR-O', 'FR-O']
    PARENT_FED = 'IPF'

    # essential columns for model and feature engineer (dots could be useful so keeping for now)
    ESSENTIAL_COLUMNS = [
        'Name', 'Date', 'Sex', 'Age', 'BodyweightKg',
        'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 
        'TotalKg', 'Dots'
    ]
'''
Event: only SBD
Equipment: Raw only
Age: drop empty fields
Division: [Open, MR-O, FR-O]
TotalKg: drop empty fields (no disqualifications)
Place: must be a number (no disqualifications)
ParentFederation: only IPF
Best3s: ensure all 3 lifts are successful (no disqualifications)
'''

def data_cleaning(df):
    cfg = DataCleaningConfig()
    print('Started data cleaning...')
    data = df[
        (df['Event'] == cfg.EVENT) & 
        (df['Equipment'] == cfg.EQUIPMENT) &
        (df['Age'].notna()) &
        (df['Division'].isin(cfg.DIVISION)) &
        (df['TotalKg'].notna()) &
        (df['Place'].str.isnumeric()) &
        (df['ParentFederation'] != cfg.PARENT_FED) &
        (df['Best3SquatKg'].notna()) &
        (df['Best3BenchKg'].notna()) &
        (df['Best3DeadliftKg'].notna()) 
    ].copy()

    data = data[cfg.ESSENTIAL_COLUMNS].copy()

    # Convert to datetime and sort by lifter and date (important for time-based feature engineering like calculating progression)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(['Name', 'Date']).reset_index(drop=True)

    data.to_csv('data/2-preprocessed/cleanIPF.csv', index=False)
    print('Successfully cleaned data and saved to "data/2-preprocessed/cleanIPF.csv"')
    return data

if __name__ == '__main__':
    file_path = pd.read_csv('data/1-raw/openpowerlifting-2025-09-27-31932eca.csv')
    data_cleaning(file_path)