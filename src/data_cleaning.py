import pandas as pd

'''
Raw data columns:
['Name', 'Sex', 'Event', 'Equipment', 'Age', 'AgeClass',
'BirthYearClass', 'Division', 'BodyweightKg', 'WeightClassKg',
'Squat1Kg', 'Squat2Kg', 'Squat3Kg', 'Squat4Kg', 'Best3SquatKg',
'Bench1Kg', 'Bench2Kg', 'Bench3Kg', 'Bench4Kg', 'Best3BenchKg',
'Deadlift1Kg', 'Deadlift2Kg', 'Deadlift3Kg', 'Deadlift4Kg',
'Best3DeadliftKg', 'TotalKg', 'Place', 'Dots', 'Wilks', 'Glossbrenner',
'Goodlift', 'Tested', 'Country', 'State', 'Federation',
'ParentFederation', 'Date', 'MeetCountry', 'MeetState', 'MeetTown',
'MeetName', 'Sanctioned']
'''

class DataCleaningConfig():
    # config constants for data cleaning
    EVENT = 'SBD'
    EQUIPMENT = 'Raw'
    DIVISION = ['Open', 'MR-O', 'FR-O'] # not including Juniors, duplicated lifter data in Open divisions
    PARENT_FED = 'IPF'

    # essential columns for model and feature engineer (dots could be useful, but ensure no data leakage)
    ESSENTIAL_COLUMNS = [
        'Name', 'Date', 'Sex', 'Age', 'BodyweightKg',
        'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 
        'TotalKg' #, 'Dots'
    ]

def select_target_data(df):
    '''
    Event: only SBD
    Equipment: Raw only
    Division: [Open, MR-O, FR-O]
    ParentFederation: only IPF
    '''

    cfg = DataCleaningConfig()
    print('Filtering to only target data (IPF, Raw, SBD)...')
    data = df[
        (df['Event'] == cfg.EVENT) & 
        (df['Equipment'] == cfg.EQUIPMENT) &
        (df['Division'].isin(cfg.DIVISION)) &
        (df['ParentFederation'] == cfg.PARENT_FED)
    ].copy()
    return data

def data_cleaning(df):
    '''
    Age: drop empty fields
    TotalKg: drop empty fields (no disqualifications)
    Place: must be a number (no disqualifications)
    Best3s: ensure all 3 lifts are successful (no disqualifications)
    '''

    cfg = DataCleaningConfig()
    print('Cleaning and preprocessing data...')
    data = df[
        (df['Age'].notna()) &
        (df['TotalKg'].notna()) &
        (df['Place'].str.isnumeric()) &
        (df['Best3SquatKg'].notna()) &
        (df['Best3BenchKg'].notna()) &
        (df['Best3DeadliftKg'].notna()) 
    ].copy()

    data = data[cfg.ESSENTIAL_COLUMNS].copy()

    # Convert to datetime and sort by lifter and date (important for time-based feature engineering like calculating progression)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(['Name', 'Date']).reset_index(drop=True)
    return data

def convert_to_csv(data, save_path):
    data.to_csv(save_path, index=False)
    print(f'Successfully cleaned data and saved to "{save_path}"')
    return data

def run(data, save_path):
    target_data = select_target_data(data)
    # clean_data = data_cleaning(target_data)
    # convert_to_csv(clean_data, save_path)
    convert_to_csv(target_data, save_path)

if __name__ == '__main__':
    raw_data = pd.read_csv('data/1-raw/openpowerlifting-2025-09-27.csv')
    # save_path = 'data/2-preprocessed/cleanIPF.csv'
    save_path = 'data/1-raw/openpowerlifting-2025-09-27-IPF-Open.csv'
    run(raw_data, save_path)