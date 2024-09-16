import pandas as pd
import joblib

def load_model(model_path='models/random_forest_model.pkl'):
    """Loads a trained model from a file."""
    model = joblib.load(model_path)
    return model


def main():
    # Load the trained model
    model = load_model()
    print("Model loaded successfully.")
    
    # Load new data for prediction (example path, update as needed)
    new_data_path = 'data/processed/X_test.csv'
    X_new = pd.read_csv(new_data_path)
    
    # Make predictions
    predictions = model.predict(X_new)
    

# if __name__ == "__main__":
#     main()
