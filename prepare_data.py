import json
import pickle

with open("intents.json") as f:
    intents = json.load(f)

texts = []
labels = []

for intent, content in intents.items():
    for pattern in content["patterns"]:
        texts.append(pattern)
        labels.append(intent)

with open("training_data.pkl", "wb") as f:
    pickle.dump((texts, labels), f)
