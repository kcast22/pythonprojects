import streamlit as st
import requests
import time

# 🎭 Set Up the Page Configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",
    layout="centered"
)

# 🎨 Custom Styling
st.markdown("""
    <style>
        body { font-family: 'Arial', sans-serif; }
        .title { color: #4A90E2; font-size: 32px; font-weight: bold; text-align: center; }
        .stTextArea textarea { font-size: 18px; }
        .stButton>button { background-color: #4CAF50; color: white; font-size: 18px; padding: 10px; border-radius: 10px; }
        .result-box { font-size: 24px; font-weight: bold; text-align: center; padding: 10px; border-radius: 10px; }
        .positive { background-color: #D4EDDA; color: #155724; }
        .negative { background-color: #F8D7DA; color: #721C24; }
    </style>
""", unsafe_allow_html=True)

# 🎭 App Title
st.markdown('<p class="title">🎭 Movie Review Sentiment Analysis</p>', unsafe_allow_html=True)

st.subheader("💡 Enter a movie review below and find out if it's Positive or Negative!")

# 📌 User Input Section
user_input = st.text_area("📝 Write your review here:", height=150)

# 🌍 FastAPI Endpoint
API_URL = "http://127.0.0.1:8000/predict"

# 🎯 Analyze Sentiment Button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        with st.spinner("🔄 Analyzing sentiment... Please wait..."):
            time.sleep(1)  # Simulate processing time
            
            # Send request to FastAPI
            response = requests.post(API_URL, json={"text": user_input})

            if response.status_code == 200:
                result = response.json()
                sentiment = result.get("sentiment", "Unknown")

                # Display Sentiment Result with Styling
                if sentiment == "positive":
                    st.markdown('<p class="result-box positive">✅ Positive Sentiment 🎉</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="result-box negative">❌ Negative Sentiment 😞</p>', unsafe_allow_html=True)
            else:
                st.error("Something went wrong! Please try again.")
    else:
        st.warning("⚠️ Please enter a review before analyzing.")

# 📌 Footer Section
st.markdown("---")
st.markdown("🚀 Built with ❤️ using **FastAPI & Streamlit** | Created by **Your Name**")
