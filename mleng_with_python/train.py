from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():
    # Load the iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a logistic regression model
    model = LogisticRegression(max_iter=200)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Evaluate the model on the testing data
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()