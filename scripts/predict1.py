import pandas as pd
import pickle
from sklearn.metrics import classification_report

def predict_churn(input_file, output_file):
    # Load test data
    test = pd.read_csv(input_file)

    # Load the trained model
    with open('C:/Users/pande/Churn-Prediction-Online-Learning/models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load the TF-IDF vectorizer
    with open('C:/Users/pande/Churn-Prediction-Online-Learning/models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    # Transform text data
    tfidf_features = tfidf.transform(test['Feedback']).toarray()

    # Combine structured and unstructured features
    X_structured = test[['Age', 'Time_Spent', 'Courses_Completed']].values
    X = pd.concat([pd.DataFrame(X_structured), pd.DataFrame(tfidf_features)], axis=1)

    # Predict probabilities and adjust the decision threshold
    probabilities = model.predict_proba(X)
    test['Churn_Probability'] = probabilities[:, 1]
    test['Predictions'] = (test['Churn_Probability'] > 0.6).astype(int)

    # Evaluate the predictions
    print("Classification Report:")
    print(classification_report(test['Is_Churn'], test['Predictions']))

    # Save predictions
    test.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}.")

if __name__ == "__main__":
    predict_churn('C:/Users/pande/Churn-Prediction-Online-Learning/data/test.csv', 'C:/Users/pande/Churn-Prediction-Online-Learning/data/predictions1.csv')
