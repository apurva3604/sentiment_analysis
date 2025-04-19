import streamlit as st
import pickle

# Load model
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  # if you're using one

st.title("🎬 Movie Review Sentiment Analysis")

review = st.text_area("Enter your movie review:")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)

        if prediction[0] == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")
