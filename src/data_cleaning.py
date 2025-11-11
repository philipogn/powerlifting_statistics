'''Event: only SBD
Equipment: Raw only
Age: drop empty fields
Division: Open only (catches all with 'O', excludes lowercase 'o' such as 'Juniors')
TotalKg: drop empty fields (no DQs)
Place: must be a number (no DQs)
ParentFederation: only IPF
Best3: ensure all 3 lifts are successful (no DQs)'''

def data_cleaning(df):
    data = df[
        (df['Event'] == 'SBD') & 
        (df['Equipment'] == 'Raw') &
        (df['Age'].notna()) &
        (df['Division'].str.contains('O', na=False)) &
        (df['TotalKg'].notna()) &
        (df['Place'].str.isnumeric()) &
        (df['ParentFederation'] == 'IPF') &
        (df['Best3SquatKg'].notna()) &
        (df['Best3BenchKg'].notna()) &
        (df['Best3DeadliftKg'].notna()) 
    ]

    # Convert to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Sort by lifter and date (important for feature engineering)
    data = data.sort_values(['Name', 'Date']).reset_index(drop=True)

    data.to_csv('../data/2-preprocessed/cleanIPF.csv', index=False)

    # print(f"Original rows: {len(df)}")
    # print(f"Filtered rows: {len(data)}")
    # print(f"Unique lifters: {data['Name'].nunique()}")
    # print(f"Male: {len(data[data['Sex']=='M'])}, Female: {len(data[data['Sex']=='F'])}")
    return data