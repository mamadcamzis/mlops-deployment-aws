from huggingface_hub import hf_hub_download, hf_hub_url, cached_download
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


REPO_ID = "electricweegie/mlewp-sklearn-wine"
FILENAME = "rfc.joblib"

def load_model():
    # Download the model file from Hugging Face Hub
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    # Load the model using joblib
    model = joblib.load(model_path)
    
    return model

def load_data():
    # Load the wine dataset
    data = load_wine()
    X = data.data
    y = data.target
    # create an array of True for 2 and False otherwise
    y = y == 2
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def predict(model, X_test):
    # Make predictions using the loaded model
    predictions = model.predict(X_test)
    print(f"Predictions: {predictions}")
    return predictions


def main():
    model = load_model()
    X_train, X_test, y_train, y_test = load_data()
    predictions = predict(model, X_test)
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
    