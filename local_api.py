import requests

# GET
r = requests.get("https://deploying-a-scalable-ml-pipeline-with.onrender.com")
print(f"Status Code: {r.status_code}")
print(f"Result: {r.json()}")

# POST
sample = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}
r = requests.post("https://deploying-a-scalable-ml-pipeline-with.onrender.com/predict/", json=sample)
print(f"Status Code: {r.status_code}")
print(f"Result: {r.json()}")
