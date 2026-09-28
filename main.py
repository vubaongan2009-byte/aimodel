from fastapi import FastAPI, Header, HTTPException, Request, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import torch
import os
import json
import numpy as np
import onnxruntime as rt
from feature_extraction import FeatureExtraction
import uvicorn
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.templating import Jinja2Templates

app = FastAPI(title="AI Spam & Bullying Detection API")
templates = Jinja2Templates(directory="template")
app.mount("/static", StaticFiles(directory="static"), name = "static")
#-----------------------------
DATABASE_URL = "sqlite:///./history.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    mode = Column(String, index=True)
    text = Column(String)
    label = Column(String)
    score = Column(Float)

Base.metadata.create_all(bind=engine)
# ----------------------------
# Load SMS Spam model
# ----------------------------
sms_model_name = "rudojr/viespam-sms-model_v2.0"
sms_tokenizer = AutoTokenizer.from_pretrained(sms_model_name)
sms_model = AutoModelForSequenceClassification.from_pretrained(sms_model_name)
sms_device = "cuda" if torch.cuda.is_available() else "cpu"
sms_model.to(sms_device)

# ----------------------------
# Load Bullying model
# ----------------------------
bully_model_name = "rudojr/phobert_bullying_model"
bully_tokenizer = AutoTokenizer.from_pretrained(bully_model_name)
bully_model = AutoModelForSequenceClassification.from_pretrained(bully_model_name)
bully_device = "cuda" if torch.cuda.is_available() else "cpu"
bully_model.to(bully_device)


#------------------------------
# Load Fake news model
#------------------------------
fake_news_model_name = "rudojr/vie_fakenews_v1.0"
fakes_news_tokenizer = AutoTokenizer.from_pretrained(fake_news_model_name)
fakes_news_model = AutoModelForSequenceClassification.from_pretrained(fake_news_model_name)
fakes_news_device = "cuda" if torch.cuda.is_available() else "cpu"
fakes_news_model.to(fakes_news_device)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả domain, có thể giới hạn ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],   # Cho phép tất cả methods: GET, POST, PUT...
    allow_headers=["*"],   # Cho phép tất cả headers
)

class InputData(BaseModel):
    mode: str  # "sms", "bullying"
    text: str


def predict_sms(text: str):
    """Dự đoán SMS spam"""
    inputs = sms_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(sms_device)

    with torch.no_grad():
        outputs = sms_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = probs.argmax().item()
        score = probs[0][predicted_class_id].item() * 100

    label_map = {0: "ham", 1: "spam"}
    return label_map[predicted_class_id], score

def predict_bullying(text: str):
    inputs = bully_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(bully_device)

    with torch.no_grad():
        outputs = bully_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = probs.argmax().item()
        score = probs[0][predicted_class_id].item() * 100

    if predicted_class_id in [1, 2]:
        label = "bullying"
    else:
        label = "normal"

    return label, score

def predict_fake_news(text: str):
    inputs = fakes_news_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(fakes_news_device)

    with torch.no_grad():
        outputs = fakes_news_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = probs.argmax().item()
        predicted_class_id = probs.argmax().item()
        score = probs[0][predicted_class_id].item() * 100

    label_map = {0: "fake", 1: "real"}
    return label_map[predicted_class_id], score


with open("model/phising_url/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

sess = rt.InferenceSession("model/phising_url/phishing_model.onnx", providers=["CPUExecutionProvider"])

input_meta = sess.get_inputs()[0]
input_name = input_meta.name
input_shape = input_meta.shape  # e.g. [None, 15]
input_type = input_meta.type
outputs_meta = sess.get_outputs()

def parse_outputs(preds):
    if len(preds) >= 2:
        labels = preds[0]
        probs = preds[1]
        label = int(labels[0]) if labels.shape[0] > 0 else int(labels)
        try:
            prob = float(probs[0][1])
        except Exception:
            prob = float(probs[0]) if probs.shape[0] > 0 else None
        return label, prob

    out0 = preds[0]
    if out0.ndim == 2 and out0.shape[1] >= 2:
        prob = float(out0[0][1])
        label = int(np.argmax(out0, axis=1)[0])
        return label, prob
    if out0.ndim == 1 and out0.dtype in [np.int64, np.int32]:
        label = int(out0[0])
        return label, None
    try:
        prob = float(out0[0])
        label = 1 if prob >= 0.7 else 0
        return label, prob
    except Exception:
        return None, None

def predict_url_onnx(url):
    fe = FeatureExtraction(url)
    features = fe.getFeaturesList()

    expected_len = len(feature_columns)
    if len(features) != expected_len:
        raise ValueError(f"Feature length mismatch: got {len(features)} but model expects {expected_len}")

    X = np.array(features, dtype=np.float32).reshape(1, -1)
    preds = sess.run(None, {input_name: X})
    label, prob = parse_outputs(preds)
    if label == 0:
        return "ham", prob*100
    elif label == 1:
        return "spam", prob*100

@app.get("/", response_class = HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html",{"request":request})

@app.post("/login")
async def login(email: str = Form(...), password: str = Form(...)):
    if email == "phanthuhien" and password == "phanthuhien123":
        return RedirectResponse(url="/history", status_code=303)
    elif email == "normal" and password == "123456":
        return RedirectResponse(url="/index", status_code=303)
    return RedirectResponse(url='/', status_code=303)

@app.get('/index', response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.get('/history', response_class=HTMLResponse)
async def history_page(request: Request):
    return templates.TemplateResponse("history.html",{"request":request})

@app.post("/predict")
def predict(data: InputData):
    db = SessionLocal()
    mode = data.mode.lower()
    text = data.text.strip()

    if not text:
        return {"error": "Text input is empty!"}
    if mode == "sms":
        label, prob = predict_sms(text)
    elif mode == "bullying":
        label, prob = predict_bullying(text)
    elif mode == "fakenews":
        label, prob = predict_fake_news(text)
    elif mode == "url":
        label, prob = predict_url_onnx(text)
    else:
        return {"error": f"Unknown mode '{data.mode}', must be one of [sms, bullying]"}

    result = {
        "mode": mode,
        "text": text,
        "label": label,
        "score": round(prob, 2),
    }

    history_item = History(
            mode=result["mode"],
            text=result["text"],
            label=result["label"],
            score=result["score"],
        )
    db.add(history_item)
    db.commit()
    db.refresh(history_item)
    db.close()
    return  result

ADMIN_API_KEY = "secret123"

@app.get("/api/history")
def get_history(x_api_key: str = Header(None)):
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Admin only")

    db = SessionLocal()
    records = db.query(History).order_by(History.timestamp.desc()).limit(100).all()
    db.close()

    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "mode": r.mode,
            "text": r.text,
            "label": r.label,
            "score": r.score,
        }
        for r in records
    ]
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080)) 
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
