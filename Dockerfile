FROM python:3.10-slim

# Avoid interactive prompts during installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopencv-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all remaining code
COPY . .

# Make setup script executable
RUN chmod +x setup.sh

# Run setup (downloads models, compiles _C)
RUN ./setup.sh

CMD ["python", "main.py", "--data_dir", "data", "--results_dir", "results", "--limit", "100", "--save_visualizations"]
