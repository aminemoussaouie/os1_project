import os
import requests
from rich.console import Console

console = Console()

def download_file(url, path):
    # Ensure the folder exists before downloading
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    console.log(f"Downloading {path}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        console.log(f"[green]Downloaded {path}[/green]")
    else:
        console.log(f"[red]Failed to download {url} (Status: {response.status_code})[/red]")

def setup():
    # 1. Download Llama-3.1-8B
    llm_url = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    llm_path = "models/llm/Meta-Llama-3.1-8B-Instruct-v0.1.Q4_K_M.gguf"
    
    if not os.path.exists(llm_path):
        download_file(llm_url, llm_path)
    else:
        console.log(f"[yellow]LLM already exists at {llm_path}[/yellow]")

    # 2. Ensure TTS directory exists
    os.makedirs("models/tts", exist_ok=True)

if __name__ == "__main__":
    setup()