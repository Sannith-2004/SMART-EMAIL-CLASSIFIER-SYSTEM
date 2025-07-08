import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"\d+", "", text)      # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_dataset(df):
    df["text"] = df["text"].astype(str).apply(clean_text)
    return df