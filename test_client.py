import requests

def chat(text):
    print(f"User: {text}")
    response = requests.post(
        "http://localhost:8000/interact/text",
        json={"text": text, "user_id": "admin"}
    )
    print(f"OS1: {response.json()['response']}")
    print("-" * 20)

if __name__ == "__main__":
    chat("Hello OS1, who are you?")
    chat("Remember that my birthday is in June.")
    chat("What did I just tell you?")