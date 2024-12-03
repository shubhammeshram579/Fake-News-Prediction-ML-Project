from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import joblib
import re
import os

# Load the trained model and TfidfVectorizer
nn = joblib.load("fake_news_LR.pkl")
tfidf = joblib.load("tfidf_vectorizer_final.pkl")

def preprocess_text(text):
    text = text.lower()
    text = re.sub("[^a-z0-9 ]", "", text)
    return text

def predict_review(text):
    text = preprocess_text(text)
    x = tfidf.transform([text]).toarray()
    prob = nn.predict(x)[0]
    if prob >= 0.5:
        return "The given News is True."
    else:
        return "The given News is Fake."

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_review(text)
        return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)), host='0.0.0.0')

# Export the app as "app" for Vercel
app = app