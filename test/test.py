import requests

sentence = "Alex and Tom expect the Tibetan leader to return"
resp = requests.post("http://localhost:5000/classify", json={"sentence": sentence})

print(resp.json())
