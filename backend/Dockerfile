FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies (needed for compiling certain python packages if necessary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies list and install
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache sparse embedding model for hybrid search
RUN python -c "from langchain_qdrant import FastEmbedSparse; FastEmbedSparse(model_name='Qdrant/bm42-all-minilm-l6-v2-attentions')"


# Copy backend codebase
COPY backend /app/backend

# Expose port
EXPOSE 8000

# Start FastAPI application using uvicorn
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]
