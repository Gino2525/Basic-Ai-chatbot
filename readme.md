#  Basic AI Chatbot using Machine Learning & NLP

This is a simple yet functional chatbot built using **Python**, **scikit-learn**, and **Flask**. The chatbot can handle user queries using intent recognition and respond with predefined responses. It serves as a lightweight AI assistant and is extendable for more advanced use cases.

---

## Features

-  Intent classification using TF-IDF and a machine learning model
-  Natural Language Processing for understanding user inputs
-  Pattern-based responses using `intents.json`
-  REST API built with Flask (`/chat` endpoint)
-  Easily extensible with new intents or integrations
-  Includes model training script for customizing the chatbot

---

## Technologies Used

- Python 3.x
- Flask (for REST API)
- scikit-learn (ML model)
- TF-IDF Vectorizer (for NLP)
- JSON (to define intents and responses)
- Pickle (for saving model and vectorizer)


## 🛠️ Getting Started

### ✅ 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/Gino2525/Basic-Ai-chatbot.git
cd Basic-Ai-chatbot

python -m venv venv
pip install -r requirements.txt
python app.py


