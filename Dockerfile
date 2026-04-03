FROM python:3.11-slim

WORKDIR /app

# Install only runtime deps (no dev tools in prod image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: show help. Override with docker run ... analyze /logs/run.log
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
