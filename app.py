from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize FastAPI app
app = FastAPI()


# Define request body schema
class Message(BaseModel):
    message: str


@app.post("/predict")
def predict_spam(data: Message):
    # Transform input message using the loaded vectorizer
    message_features = vectorizer.transform([data.message])

    # Predict spam probability
    prob = model.predict_proba(message_features)[0][1]  # Spam probability

    # Determine if the message is spam
    result = "SPAM" if prob > 0.5 else "NOT SPAM"
    confidence = round(prob * 100, 2) if prob > 0.5 else round((1 - prob) * 100, 2)

    return {"prediction": result, "confidence": f"{confidence}%"}


@app.get("/")
def root():
    return {"message": "Spam Detection API is running!"}
