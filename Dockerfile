FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for pandas and matplotlib
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libfreetype6-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip, setuptools, wheel first
RUN pip install --upgrade pip setuptools wheel

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000
EXPOSE 10000

CMD gunicorn --bind 0.0.0.0:$PORT app:app
