import requests
import json

# URL must be localhost since we are running natively in WSL
API_URL = "http://localhost:8000"

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
            # Optional: Print internal thought process if available
            if 'meta' in data:
                 print(f"[Internal State]: Confidence {data['meta'].get('confidence', 'N/A')}")
        else:
            print(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Connection Error: Is main.py running? \nDetails: {e}")
    print("-" * 20)

if __name__ == "__main__":
    chat("Hello OS1, confirm you are running on my RTX 3050.")
    chat("Remember that my birthday is in June.")
    chat("What is 25 * 4?")
