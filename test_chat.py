import requests
import json

url = "http://127.0.0.1:8000/chat"

# payload = {
#     "messages": [
#         {"role": "user", "content": "What is the difference between OPQ32r and a standard coding test?"}
#     ]
# }

payload = {
    "messages": [
        {"role": "user", "content": "I am hiring a Java developer."}, # Turn 1
        {"role": "assistant", "content": "Sure! What seniority level?"}, # AI Reply 1
        {"role": "user", "content": "Senior level, 10+ years exp."} # Turn 2
    ]
}


print(f"Sending request to {url}...")
try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
    print("Make sure the server is running (python main.py)")
