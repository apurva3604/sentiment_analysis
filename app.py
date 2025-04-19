from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return render_template('index.html', review=review, result=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
