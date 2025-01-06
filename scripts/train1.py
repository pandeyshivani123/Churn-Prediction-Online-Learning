import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Train script
def train_model():
    # Load training data
    train = pd.read_csv('C:/Users/pande/Churn-Prediction-Online-Learning/data/train.csv')

    # Feature extraction (TF-IDF for text data)
    tfidf = TfidfVectorizer(max_features=100)
    tfidf_features = tfidf.fit_transform(train['Feedback']).toarray()

    # Save the TF-IDF vectorizer
    with open('C:/Users/pande/Churn-Prediction-Online-Learning/models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    print("TF-IDF vectorizer saved successfully.")

# Execute the training function
if __name__ == "__main__":
    train_model()
