import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 1: Load Dataset
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Step 2: Label the data
fake["label"] = 0
real["label"] = 1

# Step 3: Combine datasets
data = pd.concat([fake, real])
data = data[["text", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)

# Step 4: Clean Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

data["text"] = data["text"].apply(clean_text)

# Step 5: Visualize Data Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=data)
plt.title("Label Distribution (0 = Fake, 1 = Real)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# Step 7: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 8: Model Training
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Step 9: Model Evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 10: Custom Prediction
def predict_news(text):
    clean = clean_text(text)
    vect = vectorizer.transform([clean])
    pred = model.predict(vect)[0]
    return "Real" if pred == 1 else "Fake"

# Example
sample = "The economy is showing strong signs of recovery."
print("Sample Prediction:", predict_news(sample))

# Step 11: Save Model and Vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")