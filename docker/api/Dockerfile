FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./

RUN pip install --no-cache-dir \
    ".[dense,simulator,api]"

EXPOSE 8000

CMD ["uvicorn", "supportbench.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
