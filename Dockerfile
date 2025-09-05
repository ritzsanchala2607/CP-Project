FROM python:3.10.12-slim

WORKDIR /app

# Set cache directories
ENV HF_HOME=/app/.cache
ENV MPLCONFIGDIR=/app/.cache

# Install dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && apt-get clean

# Create cache directory
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model
RUN python -c "from transformers import SamModel, SamProcessor; \
    SamModel.from_pretrained('Zigeng/SlimSAM-uniform-50', cache_dir='/app/.cache'); \
    SamProcessor.from_pretrained('Zigeng/SlimSAM-uniform-50', cache_dir='/app/.cache')"

# Copy app code
COPY . .

# Set port
ENV PORT=7860

# Run with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "2"]
