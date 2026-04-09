# 🚀 Zero-Copy ML IPC Pipeline 

A high-performance machine learning inference pipeline demonstrating the extreme latency advantages of **Zero-Copy IPC (Inter-Process Communication)** versus traditional HTTP/JSON REST APIs.

## 🎯 Architecture Overview

In distributed machine learning environments, data frequently needs to be passed between standard lightweight API gateways and heavy GPU-bound worker processes. 
- **The Traditional Problem:** Native web backends rely on JSON encoded strings. When delivering large multi-megabyte Float arrays, typical REST streams choke the CPU converting native bits to strings, copying bytes over sockets, and decoding those strings *back* into memory bits. This introduces massive **O(N)** latency bottlenecks linearly tied to the payload volume.
- **The Zero-Copy IPC Solution:** By mapping Python PyTorch/Numpy memory dynamically into **POSIX Shared Memory** spaces utilizing columnar analytics framework **Apache Arrow**, our ML pipeline parses gigantic float arrays absolutely instantly. Because the underlying hardware merely points the worker directly to the physical mapping address, network parsing is deleted entirely. The transaction costs remain at **O(1)**.

## 📊 Engineering Insights

It's extraordinarily important to highlight:
* **For Small Payloads (< 1,000 items):** REST JSON remains sufficiently fast. The fundamental microsecond cost of allocating POSIX memory bounds might even slightly outpace the JSON stringification process. 
* **For Large Payloads (> 100,000 items):** The Zero-Copy PyArrow pipeline completely crushes HTTP REST. It handles millions of numerical data elements smoothly without linear lag, making it an absolute necessity for real-time computer vision or quantitative trading ML pipelines!

## ⚙️ Setup & Deployment 

**Prerequisites:** Python 3.10+ (Tested against Python 3.13)

**1. Establish the Virtual Environment:**
```bash
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate  
# On Windows:
.\.venv\Scripts\activate
```

**2. Install runtime configurations:**
```bash
pip install -r requirements.txt
```

**3. Initialize the compute ecosystem:**
This will spawn the native ML worker alongside dual-routing API endpoints, automatically parsing safe active ports to avoid crash bindings. Keep this terminal shell active!
```bash
cd .gemini/antigravity/scratch/zero_copy_ml
python run.py
```

**4. Launch the Front-End Profile Dashboard:**
In an adjacent shell, startup the live visualization GUI analyzing the background components.
```bash
python -m streamlit run app.py
```

## 📈 Trigger Benchmarks Directly
If you want to view pure terminal CLI outputs breaking down the exact millisecond differences between networking parsing and hardware IPC transfers natively:
```bash
python -m benchmarks.run_benchmark
```

<br>

> **Developed by:** Jaya Dutta  
> **Course:** Final Year CSE (AI/ML)
