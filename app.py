from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
import json

app = FastAPI()

model_path = "data/final_salary_prediction.pkl"

class MLflowModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = mlflow.pyfunc.load_model(context.artifacts["model"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)

@app.post("/predict")
async def predict(data: dict):
    try:
        job_id = data.get("jobId")

        if job_id is None:
            raise HTTPException(status_code=400, detail="jobId is required in the input data")

        input_data = pd.DataFrame({"jobId": [job_id]})

        prediction = model.predict(None, input_data)

        predicted_salary = prediction[0]

        result = {"jobId": job_id, "predicted_salary": predicted_salary}

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    model = MLflowModel(artifact_path=model_path)  # path to model
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
