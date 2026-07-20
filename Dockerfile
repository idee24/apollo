FROM python:3.12-slim

WORKDIR /app

# Install deps first for layer caching.
COPY pyproject.toml README.md ./
COPY engine ./engine
COPY training ./training
COPY api ./api
COPY docs ./docs
COPY data/registry.yaml ./data/registry.yaml
RUN pip install --no-cache-dir ".[api]"

# Model artifacts are NOT baked in (gitignored, and data is non-redistributable).
# Mount them at runtime, e.g.:
#   docker run -p 8000:8000 -v $(pwd)/models:/app/models apollo-engine
# or point APOLLO_MODEL_PATH at a mounted file.
ENV APOLLO_RATE_LIMIT_PER_MIN=120
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
