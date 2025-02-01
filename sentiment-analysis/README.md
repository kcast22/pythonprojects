# 🎭 Sentiment Analysis API & Web App

This project is a **Sentiment Analysis system** built using **FastAPI** for the backend and **Streamlit** for the frontend. It allows users to analyze the sentiment (positive/negative) of movie reviews.

## 🚀 Features
✔️ **FastAPI** backend for sentiment prediction  
✔️ **Trained Logistic Regression model** for text classification  
✔️ **Streamlit Web UI** for easy user interaction  
✔️ **Pretrained IMDb sentiment model**  



## 🛠️ Installation & Setup

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/kcast22/pythonprojects.git
cd sentiment-analysis


## Create virtual environment

python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux


## Install dependencies **YOU WILL NEED TO USE PYTHON 13.10 or less for tensorflow to work**
pip install -r requirements.txt

#train model
python train.py

#after training the model..sentiment_model.pkl & vectorizer.pkl should now exist in models folder

##RUN APPLICATION
python -m uvicorn app:app --reload


🎯 How to Use
1️⃣ Enter a movie review in the Streamlit app
2️⃣ Click "Analyze Sentiment"
3️⃣ See if the review is Positive or Negative!

