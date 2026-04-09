import requests
from pydantic import BaseModel
from typing import List
from bs4 import BeautifulSoup

class LifterDataClass(BaseModel):
    lifter: dict
    meet_details: List[dict]

class MeetScraper():
    def __init__(self, username: str):
        self.name = username
        self.data_scrape = None
        self.response = None
        self.request_status = True
        self.meet_data = []

    def preprocess_name(self):
        self.name = self.name.replace(" ", "").lower()

    def get_request(self):
        try:
            url = f'https://www.openpowerlifting.org/u/{self.name}'
            self.response = requests.get(url)
            self.response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            self.request_status = False
            print(f'HTTP Error: {e}')
        else:
            self.data_scrape = BeautifulSoup(self.response.text, 'html.parser')

    def get_name_sex(self):
        header = self.data_scrape.find('h1')
        name = header.contents[0].get_text()
        sex = header.text.split('(')[-1].replace(')', '')
        return {'Name': name, 'Sex': sex}

    def extract_history(self):
        meet_history_table = self.data_scrape.find_all('table')[1] # two tables, second contains meet history
        keys = meet_history_table.find_all('tr')[0] # column headers
        columns = [col.text.strip() for col in keys.find_all('th')]
        seen = set()

        for row in meet_history_table.find_all('tr')[1:]:
            squats = self.attempts_to_list(row.find_all('td', class_='squat'))
            bench = self.attempts_to_list(row.find_all('td', class_='bench'))
            deadlift = self.attempts_to_list(row.find_all('td', class_='deadlift'))

            for cls in ['squat', 'bench', 'deadlift']:
                tds = row.select(f'td.{cls}')
                for td in tds[1:]:
                    td.decompose()
            row_data = [data.text.strip() for data in row.find_all('td')]
            di = dict(zip(columns, row_data))
            di['Squat'] = squats
            di['Bench'] = bench
            di['Deadlift'] = deadlift

            if '~' in di['Age']:
                di['Age'] = float(di['Age'].replace('~', '.5'))
            else:
                di['Age'] = float(di['Age'])

            # skip out duplicate entries, some lifters have dupes.
            key = (di.get('Date'), di.get('Competition'))
            if key in seen:
                continue
            seen.add(key)

            self.meet_data.append(di)
        return self.meet_data

    @staticmethod
    def attempts_to_list(attempts):
        attempts_list = []
        for lift in attempts[:3]:
            try:
                attempts_list.append(float(lift.text.strip()))
            except ValueError:
                continue
        return attempts_list

    def get_lifter_history(self) -> LifterDataClass:
        self.preprocess_name()
        self.get_request()
        lifter = self.get_name_sex()
        if self.request_status:
            self.extract_history()
        meet_data = self.meet_data[::-1]

        return LifterDataClass(lifter=lifter, meet_details=meet_data)

if __name__ == '__main__':
    scrape = MeetScraper("phillip ngo")
    scrape.get_lifter_history()
