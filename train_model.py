import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from app import clean_text

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine datasets
data = pd.concat([fake, real], ignore_index=True)

# Prepare title and article text
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

# Combine title and article
data["content"] = data["title"] + " " + data["text"]

# Keep required columns
data = data[["content", "label"]]

# Remove duplicates
data.drop_duplicates(subset="content", inplace=True)

# Clean text
data["content"] = data["content"].apply(clean_text)

# Separate features and labels
X = data["content"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
	X, y,
	test_size=0.2,
	random_state=42,
	stratify=y
)

# TF-IDF feature extraction
vectorizer = TfidfVectorizer(
	stop_words="english",
	max_df=0.7
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

# Evaluate model
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("Model Training Completed!")
print("Total News:", len(data))
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("Model Classes:", model.classes_)
print("Fake Label: 0")
print("Real Label: 1")

# Save model
with open("misinformation_model.pkl", "wb") as file:
	pickle.dump(model, file)

# Save vectorizer
with open("tfidf_vectorizer.pkl", "wb") as file:
	pickle.dump(vectorizer, file)

print("\nModel and Vectorizer Saved Successfully!")
