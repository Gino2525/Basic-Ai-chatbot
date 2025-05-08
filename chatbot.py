import pickle
import random
import json

# Load model and vectorizer
with open("intent_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load intents
with open("intents.json") as f:
    intents = json.load(f)

# Predict intent
def predict_intent(user_input):
    X = vectorizer.transform([user_input])
    intent = model.predict(X)[0]
    return intent

# Generate response
def get_response(intent):
    responses = intents[intent]["responses"]
    return random.choice(responses)

# Run chatbot
def chat():
    print("Chatbot: Hello! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        intent = predict_intent(user_input)
        response = get_response(intent)
        print("Chatbot:", response)

if __name__ == "__main__":
    chat()
