FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/default.yaml ./config/default.yaml

# Artifacts and memory are mounted as volumes
VOLUME ["/app/artifacts", "/app/memory"]

EXPOSE 8080

ENV PYTHONUNBUFFERED=1

# Default: start dashboard. Override for CLI:
# docker run ... research-evo run "your topic"
CMD ["research-evo", "serve", "--port", "8080", "--no-browser"]
