import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yaml
import joblib

def load_preprocessed_data():
    """Loads preprocessed data from CSV files."""
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()  # Convert DataFrame to Series
    y_val = pd.read_csv('data/processed/y_val.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train, y_train, n_estimators, max_depth, min_samples_split, config):
    """Trains a model using the given training data and configuration."""
    model = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_samples_split = min_samples_split
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    """Evaluates the model on the validation set."""
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
   

def main():

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

# if __name__ == "__main__":
#     main()
