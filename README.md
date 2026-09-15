# Powerlifting prediction model
- A powerlifting prediction model that predicts an existing competing powerlifter's future TotalKg based on history of the lifters meet history on the OpenPowerlifting database.
- Model trained with over 150,000 rows of lifters that competed in Raw SBD competitions in the International Powerlifting Federation

## Results from 2025-09-27 data snapshop
Model MAE is **19.1 kg** vs **23.1 kg** for simply repeating the lifter's previous total, a **17.1% reduction in error** over the persistance baseline

## Model vs baselines
|                              |     MAE |    RMSE |     R2 |
|:-----------------------------|--------:|--------:|-------:|
| Predict training mean        | 135.347 | 159.548 | -0.029 |
| Persistance (previous total) |  23.058 |  40.958 |  0.932 |
| Model                        |  19.114 |  31.912 |  0.959 |

[Further results](reports/evaluation.md)

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
- Uses `models/XGBR_model_v1.pkl` by default.
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
  --input data/1-raw/openpowerlifting-2025-09-27.csv
```

## Evaluation report
- Every training run writes `reports/evaluation.md`, which compares the model on the held-out test set against two naive baselines: predicting the training mean, and a persistence baseline (predicting the lifter's previous total).
- The report includes MAE/RMSE/R2 for each predictor and an MAE breakdown by days since last meet and by number of prior meets.

## Train with new data
- The entire pipeline can be rerun and retrained with updated data. Data is split by date into train / validation / test at the 70th and 85th quantiles hyperparameters are tuned against the validation fold, the final model is refit on train+validation, and the test fold is only used for the evaluation report.
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
