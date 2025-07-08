from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer():
    return TfidfVectorizer(stop_words='english', max_features=3000)

def extract_features(vectorizer, texts):
    return vectorizer.fit_transform(texts)