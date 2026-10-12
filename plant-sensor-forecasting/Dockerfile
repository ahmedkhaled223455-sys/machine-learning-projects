# Containerised training + backtest + promotion gate (CPU).
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-torch.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements-torch.txt
COPY src ./src
COPY tests ./tests
RUN mkdir -p results
# Fails the build if any test fails; then the default command runs the
# full pipeline and writes results/metrics.json + results/registry.json.
RUN python -m pytest -q tests
CMD ["python", "src/backtest.py"]
