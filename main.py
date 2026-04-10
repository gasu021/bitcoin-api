from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IBM_KEY   = os.environ.get("IBM_KEY", "")
DEPLOY_ID = os.environ.get("DEPLOY_ID", "964ce318-2911-4ced-a15e-c4a698923b89")

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(body: dict):
    try:
        token_r = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            data={"apikey": IBM_KEY, "grant_type": "urn:ibm:params:oauth:grant-type:apikey"}
        )
        token = token_r.json()["access_token"]
        
        from datetime import datetime
        payload = {
            "input_data": [{
                "fields": ["datetime","vader","textblob","subjectivity",
                           "retorno_1h","retorno_3h","rsi",
                           "volatilidad_6h","volumen_relativo","precio_vs_ma12"],
                "values": [[
                    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                    body["vader"], body["textblob"], body["subjectivity"],
                    body["retorno_1h"], body["retorno_3h"], body["rsi"],
                    body["volatilidad_6h"], body["volumen_relativo"], body["precio_vs_ma12"]
                ]]
            }]
        }
        
        r = requests.post(
            f"https://us-south.ml.cloud.ibm.com/ml/v4/deployments/{DEPLOY_ID}/predictions?version=2021-05-01",
            json=payload,
            headers={"Authorization": f"Bearer {token}"}
        )
        result = r.json()
        pred  = result["predictions"][0]["values"][0][0]
        probs = result["predictions"][0]["values"][0][1]
        return {
            "prediction": int(pred),
            "direction": "SUBE" if pred == 1 else "BAJA",
            "confidence": float(probs[pred]),
            "probabilities": {"baja": float(probs[0]), "sube": float(probs[1])}
        }
    except Exception as e:
        return {"error": str(e)}
