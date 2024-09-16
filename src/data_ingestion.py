import os
import zipfile
import pandas as pd

def unzip_file(zip_path, extract_to_folder):
    """Unzips a zip file into the specified folder."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)
        print(f"Extracted {zip_path} to {extract_to_folder}")

def load_data(file_path):
    """Loads data from a CSV file into a DataFrame."""
    data = pd.read_csv(file_path)
    return data

def main():
    # Define paths
    zip_file_path = 'data/raw/archive.zip'  # Update with your zip file name
    extraction_path = 'data/raw/unzipped_data/'
    extracted_file_name = 'heart-disease.csv'  # Update with the name of your CSV file inside the zip

    # Ensure extraction path exists
    os.makedirs(extraction_path, exist_ok=True)

    # Unzip the file
    unzip_file(zip_file_path, extraction_path)

    # Construct the full path to the CSV file
    csv_file_path = os.path.join(extraction_path, extracted_file_name)

    # Load the data
    data = load_data(csv_file_path)
    print("Data loaded successfully.")
    
    # For demonstration, print the first few rows of the data
    # print(data.head())

# if __name__ == "__main__":
#     main()
