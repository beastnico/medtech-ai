# Use slim Python image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (PDF, Tesseract, OpenCV, SciPy build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to get prebuilt wheels when possible
RUN pip install --upgrade pip setuptools wheel

ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV TORCH_DISABLE_JIT=1
ENV TRANSFORMERS_NO_TF=1
ENV TRANSFORMERS_NO_FLAX=1

# Copy Python dependencies first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code (but NOT large data / vectorstore / logs)
COPY app.py .
COPY config.py .
COPY core/ core/
COPY utils/ utils/
COPY scripts/ scripts/
COPY tests/ tests/

# Create directories that will be mounted
RUN mkdir -p data vectorstore logs

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
