import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_ingestion import  load_data

def preprocess_data(data):
    """Preprocess the data for training."""
    
    # Drop rows with missing target values
    data = data.dropna(subset=['target'])
    
    # Separate features and target variable
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    # Define path to the data
    
    data_file_path = 'data/raw/unzipped_data/heart-disease.csv' 
    
    # Load the data
    
    data = load_data(data_file_path)
    print("Data loaded successfully.")
    
    # Preprocess the data
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data)
    
    # Save the preprocessed data to CSV files (optional)
    
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_val.to_csv('data/processed/X_val.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_val.to_csv('data/processed/y_val.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("Data preprocessing completed and saved.")

# if __name__ == "__main__":
#     main()
