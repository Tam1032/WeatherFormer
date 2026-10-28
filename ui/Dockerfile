FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --group ui

COPY ui/ ui/
EXPOSE 7860
CMD ["/app/.venv/bin/python", "ui/app.py"]
