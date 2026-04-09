import pyarrow as pa
import torch
import logging
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared_memory.ipc import IPCManager
from models.model import get_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - WORKER - %(levelname)s - %(message)s')
logger = logging.getLogger("worker")

def deserialize_zero_copy(memory_view) -> torch.Tensor:
    """Reads arrow data directly from shared memory with STRICT zero copy!"""
    # Wrap standard PyMemoryView dynamically inside a PyArrow native buffer. ZERO allocator cost.
    buf = pa.py_buffer(memory_view)
    
    reader = pa.ipc.RecordBatchStreamReader(buf)
    batch = reader.read_next_batch()
    
    # Extract columnar array. Enforce zero_copy_only strict check! Will fail securely if memory isn't contiguous.
    np_array = batch.column(0).to_numpy(zero_copy_only=True)
    
    # PyTorch natively interprets numpy wrappers without deepcopy.
    tensor = torch.from_numpy(np_array)
    return tensor

def serialize_result(tensor: torch.Tensor) -> bytes:
    """Serializes tensor back to Arrow IPC buffer natively."""
    out_array = pa.array(tensor.detach().cpu().numpy().flatten(), type=pa.float32())
    out_batch = pa.RecordBatch.from_arrays([out_array], names=["predictions"])
    
    stream = pa.BufferOutputStream()
    with pa.ipc.RecordBatchStreamWriter(stream, out_batch.schema) as writer:
        writer.write_batch(out_batch)
    return stream.getvalue().to_pybytes()

def run_worker():
    logger.info("Initializing ML Worker Processor...")
    model = get_model()
    ipc = IPCManager(create=False)
    logger.info("Connected to Shared Memory Space via Multiprocessing ShM.")
    
    try:
        while True:
            # 1. Poll continuously and safely wait for bytes
            mem_view = ipc.wait_for_data()
            
            try:
                # 2. Immediate zero-copy pipeline (Arrow ShM -> Tensors)
                tensor = deserialize_zero_copy(mem_view)
                
                # 3. Model Inference Execution
                with torch.no_grad():
                    output = model(tensor)
                    
                # 4. Serialize to Arrow columnar stream
                res_bytes = serialize_result(output)
                
                # 5. Flush results securely back into the IPC buffer
                ipc.write_result(res_bytes)
                
            except Exception as e:
                logger.error(f"Inference crash avoided: {e}")
                
                # Avoid leaving API client deadlocked
                ipc.write_result(b"")
                
    except KeyboardInterrupt:
        logger.info("ML Worker shutting down safely.")
    finally:
        ipc.cleanup()

if __name__ == "__main__":
    run_worker()
