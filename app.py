import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit app
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("Enter a movie review below and find out whether it's **Positive** or **Negative**.")

user_input = st.text_area("✍️ Write your movie review here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review before predicting.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)
        result = "🟢 Positive Review" if prediction[0] == 1 else "🔴 Negative Review"
        st.success(f"Prediction: {result}")
