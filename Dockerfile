FROM python:3.10.12-slim

WORKDIR /app

# Set cache directories and performance optimizations
ENV HF_HOME=/tmp/.cache
ENV MPLCONFIGDIR=/tmp/.cache
ENV OMP_NUM_THREADS=4
ENV TOKENIZERS_PARALLELISM=false

# Install dependencies including CUDA support
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create upload and output directories
RUN mkdir -p /tmp/uploads /tmp/outputs && chmod -R 777 /tmp/uploads /tmp/outputs

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create temporary cache directory (will be created at runtime)
RUN mkdir -p /tmp/.cache && chmod -R 777 /tmp/.cache

# Copy app code
COPY . .

# Set port
ENV PORT=7860

# Run with Gunicorn with optimized settings
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--worker-class", "gthread", "--timeout", "300"]
