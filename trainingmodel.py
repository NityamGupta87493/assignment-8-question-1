import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from joblib import dump

# Load dataset
data = pd.read_csv('train.csv')

# Separate features and labels
X = data['content']
y = data['spam']

# Transform text data to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = vectorizer.fit_transform(X)

# Initialize classifiers
classifiers = {
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Train each classifier and save the models
for name, clf in classifiers.items():
    clf.fit(X_tfidf, y)
    # Save the model
    dump(clf, f"{name.lower().replace(' ', '_')}_model.joblib")  # Save model with lowercase name
    print(f"{name} model has been trained and saved.")

# Save the vectorizer
dump(vectorizer, 'tfidf_vectorizer.joblib')
print("TF-IDF Vectorizer has been saved.")
