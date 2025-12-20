from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open("heart_model.pkl", "rb") as f:
    pipeline = pickle.load(f)


SEX_MAP = {
    "kadın": 0,
    "erkek": 1
}

EXANG_MAP = {
    "hayır": 0,
    "evet": 1
}

FBS_MAP = {
    "hayır": 0,
    "evet": 1
}

CP_MAP = {
    "tipik anjina": 0,
    "atipik anjina": 1,
    "anjinal olmayan ağrı": 2,
    "asemptomatik": 3
}

RESTECG_MAP = {
    "normal": 0,
    "st-t dalga anormalliği": 1,
    "sol ventrikül hipertrofisi": 2
}

SLOPE_MAP = {
    "yükselen": 0,
    "düz": 1,
    "düşen": 2
}

THAL_MAP = {
    "normal": 1,
    "sabit defekt": 2,
    "tersinir defekt": 3
}



class HeartInput(BaseModel):
    age: int
    sex: str
    cp: str
    trestbps: int
    chol: int
    fbs: str
    restecg: str
    thalach: int
    exang: str
    oldpeak: float
    slope: str
    ca: int
    thal: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
     return templates.TemplateResponse("index.html", {"request": request})


# Data setinde string olması gereken datalar hazır encode edilmiş şekilde numeric idi. Burada kullanıcıdan string şeklinde alacağımız
# değerleri modelin anlayacağı numeric değerlere map ediyoruz.

@app.post("/predict")
async def predict(input: HeartInput):
        input_dict = {
            "age": input.age,
            "sex": SEX_MAP[input.sex.lower()],
            "cp": CP_MAP[input.cp.lower()],
            "trestbps": input.trestbps,
            "chol": input.chol,
            "fbs": FBS_MAP[input.fbs.lower()],
            "restecg": RESTECG_MAP[input.restecg.lower()],
            "thalach": input.thalach,
            "exang": EXANG_MAP[input.exang.lower()],
            "oldpeak": input.oldpeak,
            "slope": SLOPE_MAP[input.slope.lower()],
            "ca": input.ca,
            "thal": THAL_MAP[input.thal.lower()]
        }

        df = pd.DataFrame([input_dict])

        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0][1]

        return {
        "tahmin": "Kalp Hastalığı Riski VAR!" if prediction == 1 else "Kalp Hastalığı Riski YOK!",
        "risk_orani": round(probability * 100, 2)
        }