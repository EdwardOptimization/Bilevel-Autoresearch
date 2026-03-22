FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY src/ ./src/
COPY cli.py ./
COPY articles/ ./articles/

# Artifacts are mounted as volumes
VOLUME ["/app/artifacts"]

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
