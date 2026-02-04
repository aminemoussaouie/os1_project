# Use NVIDIA CUDA base image for RTX 3050 support
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Prevent interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install System Dependencies (Prolog, Audio, FFmpeg, Ninja)
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
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Working Directory
WORKDIR /app

# Install Python Dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Reinstall Llama-cpp with CUDA support
# Note: Using -DGGML_CUDA=on as per previous logs, but -DLLAMA_CUBLAS=on is standard for older versions.
# We stick to the one you used last which triggered the build, ensuring ninja is present.
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip3 install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# Copy Application Code
COPY . .

# Expose API Port
EXPOSE 8000

# Start Command
CMD ["python3", "main.py"]