# Use a Python 3.10 slim image for a smaller footprint
FROM python:3.10-slim

# Prevent .pyc files and enable real-time logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV and Git
# We combine these into one RUN command to reduce image layers
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install dependencies first (better caching)
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# 2. Handle the SORT dependency
# Using your fail-safe logic to ensure the tracker is available
RUN if [ ! -d "sort" ]; then git clone https://github.com/abewley/sort.git; fi

# 3. Copy project files
# We copy all folders (models, rl_env, src) into /app
COPY . /app

# 4. Entry point
# Using "python" instead of "python3" to match your previous setup
# Added --det-conf and --det-imgsz placeholders for RL optimization
CMD ["python", "main.py", "--video_path", "sample.mp4", "--det-conf", "0.88", "--det-imgsz", "736"]