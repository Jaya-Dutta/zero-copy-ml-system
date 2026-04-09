FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y gcc && apt-get clean
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Increase default SHM if needed, though docker-compose handles it
CMD ["python", "run.py"]
