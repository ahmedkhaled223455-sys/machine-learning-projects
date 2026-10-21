# Containerised training + benchmark (CPU).
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-torch.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements-torch.txt
COPY src ./src
COPY tests ./tests
RUN mkdir -p results && python -m pytest -q tests
CMD ["python", "src/train_eval.py"]
