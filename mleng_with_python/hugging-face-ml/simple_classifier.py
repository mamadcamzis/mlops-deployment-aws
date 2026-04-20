"""
Random Forest Classifier on the wine dataset.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import joblib

from sklearn.metrics import classification_report


def main():
    # Load the wine dataset
    data = load_wine()
    X = data.data
    y = data.target

    # create an array of True for 2 and False otherwise
    y = y == 2

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a random forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Evaluate the model on the testing data
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    # Save the model to a file using joblib
    joblib.dump(model, "rfc.joblib")

if __name__ == "__main__":
    main()

    
    