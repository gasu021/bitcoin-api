from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="Bitcoin Sentiment Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IBM_KEY   = os.environ.get("IBM_KEY", "")
DEPLOY_ID = os.environ.get("DEPLOY_ID", "964ce318-2911-4ced-a15e-c4a698923b89")
IBM_URL   = f"https://us-south.ml.cloud.ibm.com/ml/v4/deployments/{DEPLOY_ID}/predictions?version=2021-05-01"

def get_token():
    r = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "apikey": IBM_KEY,
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
        }
    )
    data = r.json()
    if "access_token" not in data:
        raise HTTPException(status_code=500, detail="Error obteniendo token IBM")
    return data["access_token"]

class PredictRequest(BaseModel):
    vader: float
    textblob: float
    subjectivity: float
    retorno_1h: float
    retorno_3h: float
    rsi: float
    volatilidad_6h: float
    volumen_relativo: float
    precio_vs_ma12: float

@app.get("/")
def root():
    return {"status": "ok", "message": "Bitcoin Sentiment Predictor API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(req: PredictRequest):
    if not IBM_KEY:
        raise HTTPException(status_code=500, detail="IBM_KEY no configurada")
    try:
        from datetime import datetime
        token = get_token()
        payload = {
            "input_data": [{
                "fields": [
                    "datetime", "vader", "textblob", "subjectivity",
                    "retorno_1h", "retorno_3h", "rsi",
                    "volatilidad_6h", "volumen_relativo", "precio_vs_ma12"
                ],
                "values": [[
                    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                    req.vader, req.textblob, req.subjectivity,
                    req.retorno_1h, req.retorno_3h, req.rsi,
                    req.volatilidad_6h, req.volumen_relativo, req.precio_vs_ma12
                ]]
            }]
        }
        r = requests.post(
            IBM_URL,
            json=payload,
            headers={"Authorization": f"Bearer {token}"}
        )
        result = r.json()
        if "predictions" not in result:
            raise HTTPException(status_code=500, detail=str(result))
        pred  = result["predictions"][0]["values"][0][0]
        probs = result["predictions"][0]["values"][0][1]
        return {
            "prediction": int(pred),
            "direction": "SUBE" if pred == 1 else "BAJA",
            "confidence": float(probs[pred]),
            "probabilities": {"baja": float(probs[0]), "sube": float(probs[1])}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
