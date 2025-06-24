import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load word index and reverse it for decoding
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load trained model
model = load_model("simple_rnn_imdb.h5")

# Set maxlen (same used in training)
MAX_LEN = 500


# Decode function
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])


# Preprocess text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=MAX_LEN)
    return padded_review


# Prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😞"
    return sentiment, float(prediction[0][0])


# ---------------- STREAMLIT APP UI ---------------- #
st.set_page_config(page_title="IMDB Sentiment Analyzer", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #111827;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput>div>div>input {
        background-color: #1f2937;
        color: white;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🎬 IMDB Sentiment Analyzer")
st.subheader("💬 Enter a movie review to find out if it's Positive or Negative")

review_input = st.text_area(
    "Your Review", height=150, placeholder="Type your review here..."
)

if st.button("Analyze Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter a review before submitting.")
    else:
        sentiment, score = predict_sentiment(review_input)
        st.markdown(
            f"""
            ### Prediction: **{sentiment}**
            **Confidence Score:** `{score:.2f}`
        """
        )

        if sentiment.startswith("Positive"):
            st.success("Great! It looks like a good movie review. 🍿")
        else:
            st.error("Hmm... the review seems negative. 🥀")

st.markdown("---")
st.caption("🔍 Built with TensorFlow & Streamlit | by [Your Name]")
