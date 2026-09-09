FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow's image codecs and matplotlib's font cache build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./

RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Applies migrations, then starts the API. Uses shell form so ${PORT} is honored on hosts
# (e.g. Render) that assign it dynamically, falling back to 8000 for local `docker run`.
CMD alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
