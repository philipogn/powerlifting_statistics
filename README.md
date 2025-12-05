# Powerlifting prediction model 

- A powerlifting prediction model that predicts a future TotalKg based on history of a lifters meet.
- Also able to list competition history of a lifter in JSON format
- Model trained with over 300,000 rows of lifters that competed in Raw Full Power competitions in the International Powerlifting Federation
- Predictor model deployed using FastAPI and deployed on Docker

To run the predictor, in the root directory, build the container 
``` sh
docker build -t predict-fastapi .
```

Then run 
``` sh
docker run -p 8000:8000 predict-fastapi
```
