FROM ghcr.io/astral-sh/uv:0.11-debian-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
COPY . .
RUN uv sync --no-dev \
 && useradd --create-home --shell /usr/sbin/nologin agent \
 && chown -R agent:agent /app

USER agent

CMD ["agents", "run-worker"]
