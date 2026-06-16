FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY schemas/ schemas/
COPY config.yaml* ./

# Install project dependencies
RUN pip install --no-cache-dir ".[dev]"

# Copy remaining project files
COPY . .

# Create output directories
RUN mkdir -p data cache checkpoints results notebooks

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

# Default command
CMD ["python", "train.py", "--config", "config.yaml"]
