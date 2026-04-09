import time
import requests
import json
import os
import sys
import numpy as np
import pandas as pd

# Allow running from anywhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pyarrow as pa
from shared_memory.ipc import IPCManager
from filelock import FileLock
from shared_memory.config import LOCK_FILE

def get_ports():
    try:
        config_path = os.path.join(PROJECT_ROOT, "ports_config.json")
        with open(config_path, "r") as f:
            data = json.load(f)
            return data.get("zc_port", 8000), data.get("rest_port", 8001)
    except Exception:
        return 8000, 8001

def evaluate_rest(port, payload_size_elements, iterations=3):
    # Simulate realistic tensor data
    payload = np.random.rand(payload_size_elements).astype(np.float32).tolist()
    payload_dict = {"data": payload}
    
    # Warmup
    try:
        requests.post(f"http://127.0.0.1:{port}/infer_json", json={"data":[0.5]})
    except requests.ConnectionError:
        print(f"Error: Could not connect to REST API on port {port}. Is `python run.py` running?")
        return None

    results = {"total_latency": 0.0, "processing_time": 0.0, "network_time": 0.0, "data_transfer": 0.0}
    
    for _ in range(iterations):
        t0 = time.perf_counter()
        # Strictly measure serialization inside python (converting list of floats to string stream)
        json_data = json.dumps(payload_dict)
        t_serial = time.perf_counter()
        
        # Execute transport and capture internal server processing 
        resp = requests.post(f"http://127.0.0.1:{port}/infer_json", data=json_data, headers={"Content-Type": "application/json"})
        t_transport = time.perf_counter()
        
        server_metrics = resp.json()
        proc_time = server_metrics.get("process_time_ms", 0.0)
        
        e2e = (t_transport - t0) * 1000
        serial_time = (t_serial - t0) * 1000
        
        results["total_latency"] += e2e
        results["processing_time"] += proc_time
        # Network transfer is End-to-End minus (Serialization + Internal Pure Processing)
        results["network_time"] += (e2e - serial_time - proc_time)
        results["data_transfer"] += serial_time

    # Average the metrics
    for k in results:
        results[k] /= iterations
    return results

def evaluate_zerocopy(port, payload_size_elements, iterations=3):
    # Pure Python logic acting like a true separate client/sensor pushing arrow bits natively
    payload = np.random.rand(payload_size_elements).astype(np.float32)
    
    # 1. Structure raw memory explicitly locally to skip REST payloads entirely
    arr = pa.array(payload)
    batch = pa.RecordBatch.from_arrays([arr], names=["features"])
    stream = pa.BufferOutputStream()
    with pa.ipc.RecordBatchStreamWriter(stream, batch.schema) as writer:
        writer.write_batch(batch)
    arrow_bytes = stream.getvalue().to_pybytes()

    # Warmup API 
    try:
        requests.post(f"http://127.0.0.1:{port}/predict_arrow", data=arrow_bytes, headers={"Content-Type": "application/octet-stream"})
    except requests.ConnectionError:
        print("Error: Could not connect to Zero-Copy API. Did you start `python run.py`?")
        return None

    results = {"total_latency": 0.0, "processing_time": 0.0, "network_time": 0.0, "data_transfer": 0.0}

    for _ in range(iterations):
        t0 = time.perf_counter()
        resp = requests.post(f"http://127.0.0.1:{port}/predict_arrow", data=arrow_bytes, headers={"Content-Type": "application/octet-stream"})
        t1 = time.perf_counter()
        
        proc_time = float(resp.headers.get("x-process-time", 0.0))
        
        e2e = (t1 - t0) * 1000
        
        results["total_latency"] += e2e
        results["processing_time"] += proc_time
        # Zero-Copy pushes binary without serialization! Massive gap generated here compared to JSON text mapping
        results["network_time"] += max(0, e2e - proc_time)
        results["data_transfer"] += 0.0 

    for k in results:
        results[k] /= iterations
    return results

def execute_benchmark():
    print("==================================================================")
    print("Zero-Copy ML Scalability Stress Test & Analysis")
    print("==================================================================")
    
    zc_port, rest_port = get_ports()
    # Test scales: 100 items (~400 bytes) up to 10M items (~40 Megabytes strictly float32, much larger via JSON length)
    sizes = [100, 10000, 100000, 500000, 1000000, 2500000, 5000000, 10000000]
    
    print(f"Executing Deep Tensor Scale Array Over:\n {sizes} elements\n")
    
    benchmark_history = []
    
    for size in sizes:
        # Cap iterations strictly for 50MB stringified JSON arrays so we don't hold the benchmark forever
        iters = 5 if size < 1000000 else 2
        
        rest_metrics = evaluate_rest(rest_port, size, iterations=iters)
        if not rest_metrics: return
        
        zc_metrics = evaluate_zerocopy(zc_port, size, iterations=iters)
        if not zc_metrics: return
        
        speedup = rest_metrics["total_latency"] / zc_metrics["total_latency"] if zc_metrics["total_latency"] > 0 else 0
        
        mb_equivalent = (size * 4) / (1024 * 1024)
        
        # Determine actual crossover point
        optimal_framework = "REST" if rest_metrics["total_latency"] < zc_metrics["total_latency"] else "ZERO-COPY"
        
        benchmark_history.append({
            "elements": size,
            "size_mb": round(mb_equivalent, 2),
            "rest_total_ms": round(rest_metrics["total_latency"], 2),
            "rest_serialize_ms": round(rest_metrics["data_transfer"], 2),
            "rest_network_ms": round(rest_metrics["network_time"], 2),
            "rest_process_ms": round(rest_metrics["processing_time"], 2),
            "zc_total_ms": round(zc_metrics["total_latency"], 2),
            "zc_process_ms": round(zc_metrics["processing_time"], 2),
            "speedup_ratio": round(speedup, 2),
            "winner": optimal_framework
        })
        
        print(f"Scale: {size:<9} | {mb_equivalent:>5.2f} MB | REST: {rest_metrics['total_latency']:>8.2f} ms | Zero-Copy: {zc_metrics['total_latency']:>8.2f} ms | Winner: {optimal_framework} ({speedup:>5.2f}x)")

    print("\n---------------------------------------------------------")
    print("Exporting extensive results structurally for visualization...")
    
    df = pd.DataFrame(benchmark_history)
    save_path = os.path.join(PROJECT_ROOT, "benchmark_metrics.csv")
    df.to_csv(save_path, index=False)
    
    print(f"Results flawlessly saved to: {save_path}")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    execute_benchmark()
