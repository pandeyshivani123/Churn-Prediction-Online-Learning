import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

def train_model():
    # Load data
    train = pd.read_csv('C:/Users/pande/Churn-Prediction-Online-Learning/data/train.csv')

    # Feature extraction
    tfidf = TfidfVectorizer(max_features=100)
    tfidf_features = tfidf.fit_transform(train['Feedback']).toarray()

    # Combine structured and unstructured features
    X_structured = train[['Age', 'Time_Spent', 'Courses_Completed']].values
    X = pd.concat([pd.DataFrame(X_structured), pd.DataFrame(tfidf_features)], axis=1)
    y = train['Is_Churn']

    # Train model
    model = GradientBoostingClassifier()
    model.fit(X, y)

    # Save model
    with open('C:/Users/pande/Churn-Prediction-Online-Learning/models/best_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model training complete. Model saved.")

if __name__ == "__main__":
    train_model()
