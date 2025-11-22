
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN apt-get update && apt-get install -y build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
