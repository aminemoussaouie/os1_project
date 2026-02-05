# Base Image with CUDA support for RTX 3050
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set Env to prevent Python buffering and interactive prompts
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install System Deps
# Includes Python, Prolog, Audio libs, and Git
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    swi-prolog \
    ffmpeg \
    libportaudio2 \
    libsndfile1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python Deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Reinstall llama-cpp for CUDA (Crucial for GPU speed)
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip3 install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# Install Piper (TTS Binary) inside Docker
RUN wget -O piper.tar.gz https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz && \
    tar -xvf piper.tar.gz && \
    rm piper.tar.gz

# Copy Source Code
COPY . .

# Expose API Port
EXPOSE 8000

# Entrypoint
CMD ["python3", "main.py"]