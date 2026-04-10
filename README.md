# Powerlifting prediction model

- A powerlifting prediction model that predicts an existing competing powerlifter's future TotalKg based on history of the lifters meet history on the OpenPowerlifting database.
- Model trained with over 300,000 rows of lifters that competed in Raw SBD competitions in the International Powerlifting Federation

## Streamlit UI for non-technical users
- First clone the repo and install the requirements
```sh
pip install -r requirements.txt
```
- Then run the command for the Streamlit UI
```sh
streamlit run streamlit_app.py
```
- Enter OpenPowerlifting username, current age, and bodyweight.
- Uses `models/XGBR_model.pkl` by default.
- Returns predicted next total and improvement vs latest meet.

## Predictor model deployed using FastAPI and containerised on Docker
- Also able to list competition history of a lifter in JSON format, from the /competitions/{name} endpoint
- To run the predictor, in the root directory, build the container 
```sh
docker build -t predict-fastapi .
```

- Then run 
```sh
docker run -p 8000:8000 predict-fastapi
```

## Quickstart (recommended defaults):
- Runs the entire default pipeline with default data from 2025-09-27
```sh
python entrypoint/train_pipeline.py \
  --input data/1-raw/openpowerlifting-2025-09-27-IPF.csv
```

## Train with new data
- The entire pipeline can be rerun and retrained with updated data, the train/test splits at the 80th quantile by date (80/20 train test split)
- First clone the repo and in the root directory install the requirements
```sh
pip install -r requirements.txt
```

- Download the lastest OpenPowerlifting dataset and paste into the directory, then run with the following command with the path to the raw csv
```sh
python entrypoint/train_pipeline.py --input {path_to_raw_csv}
```

- Run with an optional custom model output path:
```sh
python entrypoint/train_pipeline.py \
  --input /path/to/new_data.csv \
  --model-output models/new_model.pkl
```

- Keep every run (timestamped filenames, no overwrite):
```sh
python entrypoint/train_pipeline.py \
  --input /path/to/new_data.csv \
  --version-outputs
```
