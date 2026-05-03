FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/hf_cache

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create HF cache directory
RUN mkdir -p $HF_HOME && chmod 777 $HF_HOME

# Install python dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
# Only copy necessary directories to keep image lean
COPY app ./app
COPY data ./data

# Expose port
EXPOSE 8000

# Run the application
# Use JSON syntax for CMD
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

