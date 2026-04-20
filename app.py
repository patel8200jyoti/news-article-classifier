from flask import Flask, request, render_template
import pickle
import re
from nltk.corpus import stopwords

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        user_text = request.form["news"]

        if len(user_text.strip()) == 0:
            prediction = "Please enter some text."

        elif len(user_text.split()) < 4:
            prediction = "Please enter a valid text."

        else:
            cleaned = clean_text(user_text)
            vect = vectorizer.transform([cleaned])

            probs = model.predict_proba(vect)[0]   # get first row directly
            confidence = max(probs)
            predicted_class = model.classes_[probs.argmax()]

            if confidence < 0.30:
                prediction = "Please enter a valid news article."
            else:
                prediction = predicted_class

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

