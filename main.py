from data_loader import load_data
from preprocess import preprocess_dataset
from feature_extraction import get_vectorizer, extract_features
from model import train_model
from predict import classify_email

def main():
    df = load_data("spam_data.csv")
    df = preprocess_dataset(df)

    vectorizer = get_vectorizer()
    X = extract_features(vectorizer, df["text"])
    y = df["label"]

    model = train_model(X, y)

    test_input = "Congratulations! You've won a free trip. Click to claim."
    result, confidence = classify_email(model, vectorizer, test_input)
    print(f"Predicted: {result}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()