from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Load trained model & vectorizer
model = pickle.load(open("models/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(review: Review):
    processed_text = review.text.lower()
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    sentiment = "positive" if prediction == 1 else "negative"
    return {"sentiment": sentiment}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
