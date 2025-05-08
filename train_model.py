import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


with open("training_data.pkl", "rb") as f:
    texts, labels = pickle.load(f)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)


model = MultinomialNB()
model.fit(X, labels)


with open("intent_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
