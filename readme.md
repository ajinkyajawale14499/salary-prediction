# Salary Prediction ML System

### To install and run on local for development
```
docker build -t salary-prediction:latest .
docker-compose up -d
```

### Local MLflow serve
mlflow server --host 127.0.0.1 --port 8080
mlflow serve salary-prediction
###

###  To stop/remove the local deployment
```
docker-compose down
```

### Example prediction 
```
{
    "jobId": "JOB1362685407692"
}
```# salary-prediction
