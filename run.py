import multiprocessing as mp
import subprocess
import time
import os
import json
import socket
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared_memory.ipc import IPCManager
from shared_memory.config import LOCK_FILE

def get_free_port(start_port):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                port += 1

def get_env_with_pythonpath():
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return env

def start_worker():
    subprocess.run([sys.executable, "-m", "worker.main"], cwd=PROJECT_ROOT, env=get_env_with_pythonpath())

def start_api(port):
    subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port)], cwd=PROJECT_ROOT, env=get_env_with_pythonpath())

def start_rest_api(port):
    subprocess.run([sys.executable, "-m", "uvicorn", "benchmarks.rest_api:app", "--host", "127.0.0.1", "--port", str(port)], cwd=PROJECT_ROOT, env=get_env_with_pythonpath())

def main():
    print("====================================")
    print("Zero-Copy ML System Initialization..")
    print("====================================")

    zc_port = get_free_port(8000)
    rest_port = get_free_port(zc_port + 1)

    with open(os.path.join(PROJECT_ROOT, "ports_config.json"), "w") as f:
        json.dump({"zc_port": zc_port, "rest_port": rest_port}, f)

    print(f"-> Assigned Zero-Copy API Port: {zc_port}")
    print(f"-> Assigned Traditional REST API Port: {rest_port}")

    # 1. Provide safe centralized shared memory allocation
    ipc = IPCManager(create=True)
    
    # 2. Reset system locks
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

    # 3. Launch isolated environment processes
    print("-> Spawning PyArrow ShM Worker...")
    worker_proc = mp.Process(target=start_worker)
    worker_proc.start()
    
    print(f"-> Spawning FastAPI Zero-Copy Gateway ({zc_port})...")
    api_proc = mp.Process(target=start_api, args=(zc_port,))
    api_proc.start()

    print(f"-> Spawning Traditional REST Model Service ({rest_port})...")
    rest_proc = mp.Process(target=start_rest_api, args=(rest_port,))
    rest_proc.start()

    try:
        time.sleep(3)
        print("\n\nAll services operational!")
        print("To observe the latency reduction benchmark, run:")
        print("    python -m benchmarks.run_benchmark")
        print("\nPress Ctrl+C to terminate the ecosystem.\n")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down compute clusters...")
        
        worker_proc.terminate()
        api_proc.terminate()
        rest_proc.terminate()
        
        worker_proc.join()
        api_proc.join()
        rest_proc.join()
        
        print("Purging PyArrow Shared Memory Allocations...")
        ipc.cleanup()
        
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            
        print("Graceful exit complete.")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
