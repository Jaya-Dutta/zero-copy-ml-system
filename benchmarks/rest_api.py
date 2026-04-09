from fastapi import FastAPI
from pydantic import BaseModel
import torch
import uvicorn
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from models.model import get_model

app = FastAPI(title="Traditional REST Worker")
model = get_model()

class InferenceRequest(BaseModel):
    data: list[float]

import time

@app.post("/infer_json")
def infer(request: InferenceRequest):
    t0 = time.perf_counter()
    # Deserialization from JSON via pydantic happens implicitly, which is typical but slow!
    tensor = torch.tensor(request.data, dtype=torch.float32)
    with torch.no_grad():
        out = model(tensor)
    preds = out.numpy().flatten().tolist()
    process_time_ms = (time.perf_counter() - t0) * 1000
    return {"predictions": preds, "process_time_ms": process_time_ms}

if __name__ == "__main__":
    uvicorn.run("benchmarks.rest_api:app", host="127.0.0.1", port=8001, log_level="error")
