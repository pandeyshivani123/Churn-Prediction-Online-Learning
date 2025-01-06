import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data():
    # Load structured data
    structured = pd.read_csv('C:/Users/pande/Churn-Prediction-Online-Learning/data/structured_data.csv')
    structured.head(3)
    
    # Load unstructured data
    unstructured = pd.read_csv('C:/Users/pande/Churn-Prediction-Online-Learning/data/unstructured_data.csv')
    unstructured.head(2)
    # Combine data
    data = pd.merge(structured, unstructured, on='User_ID')
    print(data)
    # Split into train and test sets
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # Save processed data
    train.to_csv('C:/Users/pande/Churn-Prediction-Online-Learning/data/train.csv', index=False)
    test.to_csv('C:/Users/pande/Churn-Prediction-Online-Learning/data/test.csv', index=False)

if __name__ == "__main__":
    preprocess_data()
