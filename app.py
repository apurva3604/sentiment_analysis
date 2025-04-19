from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the trained model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    # Transform the review using the vectorizer
    review_vectorized = vectorizer.transform([review])
    # Predict sentiment using the loaded model
    prediction = model.predict(review_vectorized)

    # Return the prediction to the user
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return render_template('index.html', prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
