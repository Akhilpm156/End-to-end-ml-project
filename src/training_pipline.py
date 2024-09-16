from src.data_ingestion import load_data, unzip_file
from src.data_preprocessing import preprocess_data
from src.model_training import load_preprocessed_data, train_model, evaluate_model
import os
import yaml
import joblib

def training():

    # Define paths
    zip_file_path = 'data/raw/archive.zip'  # Update with your zip file name
    extraction_path = 'data/raw/unzipped_data/'
    extracted_file_name = 'heart-disease.csv'
    
    # Ensure extraction path exists
    os.makedirs(extraction_path, exist_ok=True)

    # Unzip the file
    unzip_file(zip_file_path, extraction_path)
 
    # Construct the full path to the CSV file
    csv_file_path = os.path.join(extraction_path, extracted_file_name)

    # Load the data
    data = load_data(csv_file_path)
    print("Data loaded successfully.")

    # Define path to the data
    data_file_path = 'data/raw/unzipped_data/heart-disease.csv'  # Update with the actual CSV file path

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

    # Load preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    print("Preprocessed data loaded successfully.")
    
    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract hyperparameters
    n_estimators = int(config['model']['hyperparameters']['n_estimators'])
    max_depth = int(config['model']['hyperparameters']['max_depth'])
    min_samples_split = int(config['model']['hyperparameters']['min_samples_split'])

    print(f'current trained n_estimators {n_estimators}')
    print(f'current trained max_depth {max_depth}')
    print(f'current trained min_samples_split {min_samples_split}')

    # Train the model
    model = train_model(X_train, y_train, n_estimators, max_depth, min_samples_split, config)
    
    # Save the trained model
    joblib.dump(model, 'models/random_forest_model.pkl')
    
    print("Model saved successfully.")
    
    # Evaluate the model
    evaluate_model(model, X_val, y_val)
