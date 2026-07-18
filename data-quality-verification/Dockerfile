FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Generate the dataset, then run the verification pipeline.
# data/ and results/ are expected to be volume-mounted (see compose file)
# so artifacts persist on the host.
CMD ["sh", "-c", "python src/generate_dataset.py && python src/run_pipeline.py"]
