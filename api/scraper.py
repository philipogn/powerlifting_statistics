import io
from functools import lru_cache
from typing import List
import pandas as pd
import requests
from pydantic import BaseModel

class LifterDataClass(BaseModel):
    lifter: dict
    meet_details: List[dict]


# kept in each meet dict, Best3*/TotalKg match the training data schema exactly, the rest are for display (API responses, streamlit history table).
MEET_COLUMNS = ['Date', 'MeetName', 'Age', 'BodyweightKg',
                'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg', 'Place']

@lru_cache(maxsize=128)
def _fetch_lifter_csv(username: str) -> str:
    '''
    Lifter CSV from OpenPowerlifting (the websites own "Download as CSV" endpoint).
    Cached per process so repeated lookups of the same lifter only hit the site once
    '''
    response = requests.get(
        f'https://www.openpowerlifting.org/api/liftercsv/{username}',
        timeout=15,
        headers={'User-Agent': 'powerlifting-statistics (personal project: next-total prediction)'},
    )
    if response.status_code == 404:
        raise ValueError(f'No lifter "{username}" found on OpenPowerlifting')
    response.raise_for_status()
    return response.text


def parse_lifter_csv(csv_text: str, username: str='') -> LifterDataClass:
    '''
    Filters a lifter's full record down to what the model was trained on:
    Raw SBD meets with a real total and no DQ. Returned meets are chronological and deduplicated
    '''
    df = pd.read_csv(io.StringIO(csv_text))
    if df.empty:
        raise ValueError(f'No competition history found for "{username}"')

    meets = df[
        (df['Event'] == 'SBD') &
        (df['Equipment'] == 'Raw') &
        (df['TotalKg'] > 0) &
        (df['Place'].astype(str).str.isnumeric())
    ].copy()
    meets = meets.drop_duplicates(subset=['Date', 'MeetName'], keep='first')
    meets = meets.sort_values('Date')

    lifter = {'Name': df['Name'].iloc[0], 'Sex': df['Sex'].iloc[0]}
    return LifterDataClass(lifter=lifter, meet_details=meets[MEET_COLUMNS].to_dict('records'))


class MeetScraper():
    def __init__(self, username: str):
        self.name = username

    def preprocess_name(self):
        self.name = self.name.replace(' ', '').lower()

    def get_lifter_history(self) -> LifterDataClass:
        self.preprocess_name()
        return parse_lifter_csv(_fetch_lifter_csv(self.name), username=self.name)

if __name__ == '__main__':
    scrape = MeetScraper("phillip ngo")
    print(scrape.get_lifter_history())
