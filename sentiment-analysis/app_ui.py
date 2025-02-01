import streamlit as st
import requests

st.title("🎭 Sentiment Analysis")

text = st.text_area("Enter a review:")

if st.button("Analyze Sentiment"):
    response = requests.post("http://127.0.0.1:8000/predict", json={"text": text})
    sentiment = response.json()["sentiment"]
    st.write(f"Sentiment: **{sentiment}**")

