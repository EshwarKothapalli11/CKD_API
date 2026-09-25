from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import time

model = joblib.load("model/ckd_model.joblib")

app = FastAPI(
    title="CKD Prediction API",
    version="1.0"
)


class CKDInput(BaseModel):
    age: float = Field(..., gt=0, le=120)
    bp: float = Field(..., gt=0, le=250)
    sg: float = Field(..., ge=1.0, le=1.05)
    al: float = Field(..., ge=0, le=5)
    su: float = Field(..., ge=0, le=5)
    bgr: float = Field(..., ge=0)
    bu: float = Field(..., ge=0)
    sc: float = Field(..., ge=0)
    sod: float = Field(..., ge=0)
    pot: float = Field(..., ge=0)
    hemo: float = Field(..., ge=0)
    pcv: float = Field(..., ge=0)
    wc: float = Field(..., ge=0)
    rc: float = Field(..., ge=0)

    rbc: str = "normal"
    pc: str = "normal"
    pcc: str = "notpresent"
    ba: str = "notpresent"
    htn: str = "no"
    dm: str = "no"
    cad: str = "no"
    appet: str = "good"
    pe: str = "no"
    ane: str = "no"


total_requests = 0
successful_predictions = 0
failed_requests = 0
total_response_time = 0.0


@app.get("/")
def home():
    return {
        "service": "CKD Prediction API",
        "version": "1.0",
        "status": "running",
        "description": "REST API for Chronic Kidney Disease prediction",
        "documentation": "/docs"
    }


@app.get("/frontend")
def frontend():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "CKD Prediction API"
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": "CKD Random Forest Classifier",
        "version": "1.0",
        "algorithm": "Random Forest",
        "number_of_features": 24,
        "accuracy": 0.9643,
        "precision": 0.9459,
        "recall": 1.0,
        "f1_score": 0.9722
    }


@app.post("/predict")
def predict(data: CKDInput):
    global total_requests, successful_predictions
    global failed_requests, total_response_time

    start_time = time.time()
    total_requests += 1

    try:
        input_data = pd.DataFrame([data.model_dump()])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0].max()

        result = "CKD" if prediction == "ckd" else "Not CKD"

        successful_predictions += 1

        response_time = time.time() - start_time
        total_response_time += response_time

        return {
            "prediction": result,
            "model_output_probability": round(float(probability), 4),
            "model_version": "1.0",
            "response_time_seconds": round(response_time, 6),
            "note": "This is a machine learning model output and not a medical diagnosis."
        }

    except Exception as e:
        failed_requests += 1

        return {
            "error": "Prediction failed",
            "details": str(e)
        }


@app.get("/metrics")
def metrics():
    average_response_time = (
        total_response_time / successful_predictions
        if successful_predictions > 0 else 0
    )

    return {
        "total_requests": total_requests,
        "successful_predictions": successful_predictions,
        "failed_requests": failed_requests,
        "average_response_time_seconds": round(
            average_response_time, 6
        )
    }


@app.get("/ready")
def ready():
    return {
        "ready": model is not None,
        "status": "ready" if model is not None else "not_ready"
    }