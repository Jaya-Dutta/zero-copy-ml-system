import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pyarrow as pa
from filelock import FileLock
import os
import sys

# Add project root to sys.path so 'shared_memory' can be resolved regardless of execution directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared_memory.ipc import IPCManager
from shared_memory.config import LOCK_FILE

from contextlib import asynccontextmanager

ipc = None
lock = FileLock(LOCK_FILE)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ipc
    # Startup check
    ipc = IPCManager(create=False)
    yield
    # Shutdown cleanup
    if ipc:
        ipc.cleanup()

app = FastAPI(title="Zero-Copy ML Gateway", lifespan=lifespan)

class PredictRequest(BaseModel):
    data: list[float]

def serialize_to_arrow(data: list[float]) -> bytes:
    # Build Arrow format natively (columnar layout)
    array = pa.array(data, type=pa.float32())
    batch = pa.RecordBatch.from_arrays([array], names=["features"])
    
    stream = pa.BufferOutputStream()
    with pa.ipc.RecordBatchStreamWriter(stream, batch.schema) as writer:
        writer.write_batch(batch)
    return stream.getvalue().to_pybytes()

def deserialize_from_arrow(result_bytes: bytes) -> list[float]:
    reader = pa.ipc.RecordBatchStreamReader(result_bytes)
    batch = reader.read_next_batch()
    return batch.column(0).to_pylist()

import time

def sync_inference(data_list: list[float]):
    t0 = time.perf_counter()
    # 1. API: Transform REST data (List/JSON based) to Arrow Columnar
    arrow_bytes = serialize_to_arrow(data_list)
    
    # 2. Concurrency/Synchronization
    with lock:
        ipc.write_data(arrow_bytes)
        result_bytes = ipc.wait_for_result()
        
    preds = deserialize_from_arrow(result_bytes)
    process_time_ms = (time.perf_counter() - t0) * 1000
    return preds, process_time_ms

from fastapi.responses import Response
from fastapi import Request

@app.post("/predict_arrow")
async def predict_arrow(request: Request):
    """Pure Zero-Copy Endpoint: Accepts raw arrow bytes directly handling no string conversions natively."""
    arrow_bytes = await request.body()
    t0 = time.perf_counter()
    
    loop = asyncio.get_running_loop()
    
    def inference_thread():
        with lock:
            ipc.write_data(arrow_bytes)
            return ipc.wait_for_result()
            
    result_bytes = await loop.run_in_executor(None, inference_thread)
    process_time_ms = (time.perf_counter() - t0) * 1000
    
    return Response(content=result_bytes, media_type="application/octet-stream", headers={"x-process-time": str(process_time_ms)})
