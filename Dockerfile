FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY core/ ./core/
COPY domains/ ./domains/
COPY articles/ ./articles/

# Artifacts are mounted as volumes
VOLUME ["/app/artifacts"]

ENV PYTHONUNBUFFERED=1

RUN useradd -m -s /bin/bash appuser
USER appuser

ENTRYPOINT ["python", "-m", "domains.article_opt.cli"]
CMD ["--help"]
