import os
import requests
from rich.console import Console

console = Console()

def download_file(url, path):
    console.log(f"Downloading {path}...")
    response = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    console.log(f"[green]Downloaded {path}[/green]")

def setup():
    # 1. Download Llama-3.1-8B (Quantized for 8GB VRAM)
    llm_url = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    if not os.path.exists("models/llm/Meta-Llama-3.1-8B-Instruct-v0.1.Q4_K_M.gguf"):
        download_file(llm_url, "models/llm/Meta-Llama-3.1-8B-Instruct-v0.1.Q4_K_M.gguf")

    # 2. Piper TTS is handled dynamically, but we ensure dir exists
    os.makedirs("models/tts", exist_ok=True)

if __name__ == "__main__":
    setup()