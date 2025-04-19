import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Use your CSV path
csv_path = r"C:\Users\RAJIV\Downloads\archive (13)\IMDB Dataset.csv"
df = pd.read_csv(csv_path)

# Convert sentiment labels to 1 (positive) and 0 (negative)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model and vectorizer
joblib.dump(model, 'model/sentiment_model.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')

print("✅ Model and vectorizer saved successfully!")
