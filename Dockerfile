FROM python:3.12-slim

# Prevent interactive prompts during apt installs and buffer flushing
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies required for compilation and optional SQLite/DB drivers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlcipher-dev \
    sqlcipher \
    libpq-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install core project dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install optional SQLite-compatible drivers matching CI matrix
RUN pip install --no-cache-dir apsw || true && \
    (pip install --no-cache-dir sqlcipher3-binary || pip install --no-cache-dir sqlcipher3 || true) && \
    (pip install --no-cache-dir libsql-experimental || pip install --no-cache-dir libsql || true)

# Copy workspace source files
COPY . .

# Default entrypoint runs the Docker test matrix across live engine containers
CMD ["python3", "run_docker_tests.py"]
