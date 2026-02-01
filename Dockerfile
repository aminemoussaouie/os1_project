# Use NVIDIA CUDA base image for RTX 3050 support
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install System Dependencies (Prolog, Audio, FFmpeg)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    swi-prolog \
    ffmpeg \
    libportaudio2 \
    libsndfile1 \
    libgl1-mesa-glx \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Working Directory
WORKDIR /app

# Install Python Dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Reinstall Llama-cpp with CUDA support
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip3 install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# Copy Application Code
COPY . .

# Expose API Port
EXPOSE 8000

# Start Command
CMD ["python3", "main.py"]