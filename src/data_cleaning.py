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
    DIVISION = ['Open', 'MR-O', 'FR-O', 'Juniors', 'MR-Jr', 'FR-Jr', 'Sub-Juniors',
                'Masters 1', 'Masters 2', 'Masters 3', 'Masters 4', 'Masters 5'
                ]
    PARENT_FED = 'IPF'

    # essential columns for model and feature engineer (dots could be useful, but ensure no data leakage)
    ESSENTIAL_COLUMNS = [
        'Name', 'Date', 'Sex', 'Age', 'BodyweightKg',
        # 'Squat1Kg', 'Squat2Kg', 'Squat3Kg',
        # 'Bench1Kg', 'Bench2Kg', 'Bench3Kg', 
        # 'Deadlift1Kg', 'Deadlift2Kg', 'Deadlift3Kg', 
        'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 
        'TotalKg' #, 'Dots'
    ]

def select_target_data(df):
    '''
    Event: only SBD
    Equipment: Raw only
    Division: all common/important divisions, filtering out superfluous divs
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


def remove_duplicate_entries(df):
    # prioritise these divisions
    prioritise_divisions = ['Sub-Juniors', 'Juniors', 'MR-Jr', 'FR-Jr',
                            'Masters 1', 'Masters 2', 'Masters 3', 'Masters 4', 'Masters 5'
                            ] 

    # checking columns of duplicate values, should just work with 'Name' and 'Date'
    duplicate_cols = ['Name', 'Date', 'TotalKg']

    df['is_junior'] = df['Division'].isin(prioritise_divisions).astype(int)
    df['priority'] = df['is_junior'].apply(lambda x: 1 if x == 1 else 2)

    df_clean = df.sort_values('priority').drop_duplicates(
        subset=duplicate_cols,
        keep='first'
    ).drop(columns=['is_junior', 'priority'])

    return df_clean


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
    clean_dupes = remove_duplicate_entries(target_data)
    clean_data = data_cleaning(clean_dupes)
    convert_to_csv(clean_data, save_path)

if __name__ == '__main__':
    raw_data = pd.read_csv('data/1-raw/openpowerlifting-2025-09-27.csv')
    # save_path = 'data/2-preprocessed/cleanIPF.csv'
    save_path = 'data/2-preprocessed/openpowerlifting-IPF-clean.csv'
    run(raw_data, save_path)