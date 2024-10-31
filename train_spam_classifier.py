from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from joblib import dump
import pandas as pd

# Load the dataset with the correct column names
data = pd.read_csv("./spam_dataset.csv", encoding="ISO-8859-1")

# Rename columns for clarity (optional but helpful)
data.columns = ["label", "text",]

# Select only the necessary columns
texts = data["text"]
labels = data["label"]

# Create a model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(texts, labels)

# Save the model
dump(model, "spam_model.joblib")
