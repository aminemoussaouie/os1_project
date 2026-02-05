import requests
import json
import base64
import os
import subprocess

API_URL = "http://localhost:8000"

def play_audio(file_path):
    """Plays audio using Sox (play command)"""
    try:
        # Use 'play' from the sox package
        print(f"[Audio System] Attempting to play: {file_path}")
        subprocess.run(["play", "-q", file_path], check=True)
    except Exception as e:
        print(f"[Audio Error] Could not play sound: {e}")

def chat(text):
    print(f"User: {text}")
    try:
        response = requests.post(
            f"{API_URL}/interact/text",
            json={"text": text, "user_id": "admin"},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"OS1: {data['response']}")
            
            # Check for audio path in response (if server supports returning file path)
            # OR logic to fetch audio if your API was returning it.
            
            # Since the current main.py returns "audio_path": "output.wav"
            # We need to grab that file. 
            
            # In a real web app, the browser downloads it. 
            # Here, since client and server share the same folder (localhost), we can just play it!
            
            if 'audio_path' in data:
                 print(f"[Playing Audio]: {data['audio_path']}")
                 play_audio(data['audio_path'])
            
            if 'meta' in data:
                 print(f"[Internal State]: Confidence {data['meta'].get('confidence', 'N/A')}")
        else:
            print(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Connection Error: Is main.py running? \nDetails: {e}")
    print("-" * 20)

if __name__ == "__main__":
    chat("Hello OS1, confirm you are running on my RTX 3050.")