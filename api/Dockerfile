FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first, so this layer is cached until uv.lock changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --group api

COPY weatherformer/ weatherformer/
COPY api/ api/
RUN uv sync --frozen --group api

EXPOSE 8080
CMD ["/app/.venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
