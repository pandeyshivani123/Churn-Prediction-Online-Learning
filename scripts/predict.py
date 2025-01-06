import pandas as pd
import pickle

def predict_churn(input_file, output_file):
    # Load test data
    test = pd.read_csv(input_file)

    # Load the trained model
    with open('C:/Users/pande/Churn-Prediction-Online-Learning/models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load the saved TF-IDF vectorizer
    with open('C:/Users/pande/Churn-Prediction-Online-Learning/models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    # Transform the text data using the loaded vectorizer
    tfidf_features = tfidf.transform(test['Feedback']).toarray()

    # Combine structured and unstructured features
    X_structured = test[['Age', 'Time_Spent', 'Courses_Completed']].values
    X = pd.concat([pd.DataFrame(X_structured), pd.DataFrame(tfidf_features)], axis=1)

    # Predict
    predictions = model.predict(X)
    test['Predictions'] = predictions

    # Save results
    test.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}.")

if __name__ == "__main__":
    predict_churn('C:/Users/pande/Churn-Prediction-Online-Learning/data/test.csv', 'C:/Users/pande/Churn-Prediction-Online-Learning/data/predictions.csv')
