def classify_email(model, vectorizer, text):
    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)[0]
    confidence = model.predict_proba(transformed).max()
    return prediction, confidence