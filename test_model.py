import pandas as pd
from joblib import load

# Load the test dataset
test_data = pd.read_csv('test.csv')

# Separate features from the test dataset
X_test = test_data['content']
y_test = test_data['spam']  # Assuming you have the labels in test.csv for evaluation

# Load the TF-IDF vectorizer
vectorizer = load('tfidf_vectorizer.joblib')

# Transform the test data using the loaded vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# Load classifiers
classifiers = {
    "Naive Bayes": load('naive_bayes_model.joblib'),
    "Support Vector Machine": load('support_vector_machine_model.joblib'),
    "Random Forest": load('random_forest_model.joblib'),
    "K-Nearest Neighbors": load('k-nearest_neighbors_model.joblib'),
    "Logistic Regression": load('logistic_regression_model.joblib')
}

# Predict using each classifier and store the results
results = {}
for name, clf in classifiers.items():
    # Make predictions on the test set
    y_pred = clf.predict(X_test_tfidf)
    
    # Store the predictions
    results[name] = y_pred

    # Print some sample predictions
    print(f"Predictions from {name} model:")
    print(y_pred[:10])  # Print first 10 predictions

# Optionally, evaluate accuracy if you have labels
for name, predictions in results.items():
    accuracy = (predictions == y_test).mean()
    print(f"{name} model accuracy: {accuracy:.4f}")
