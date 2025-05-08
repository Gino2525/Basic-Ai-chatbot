from flask import Flask, request, jsonify, render_template
import pickle
import json
import random

# Load model and vectorizer
with open("intent_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load intents
with open("intents.json") as f:
    intents = json.load(f)

app = Flask(__name__)

def predict_intent(user_input):
    X = vectorizer.transform([user_input])
    intent = model.predict(X)[0]
    return intent

def get_response(intent):
    responses = intents[intent]["responses"]
    return random.choice(responses)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    intent = predict_intent(user_message)
    response = get_response(intent)

    return jsonify({"intent": intent, "response": response})

if __name__ == "__main__":
    app.run(debug=True)
