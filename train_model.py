import pandas as pd
import joblib
from preprocess import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

# Add labels
true_df["label"] = 1   # Real news
fake_df["label"] = 0   # Fake news

# Combine title + text
true_df["content"] = true_df["title"].astype(str) + " " + true_df["text"].astype(str)
fake_df["content"] = fake_df["title"].astype(str) + " " + fake_df["text"].astype(str)

# Keep required columns
true_df = true_df[["content", "label"]]
fake_df = fake_df[["content", "label"]]

# Merge and shuffle
df = pd.concat([true_df, fake_df], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean text
df["content"] = df["content"].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Vectorizer (important for accuracy)
vectorizer = TfidfVectorizer(
    max_df=0.7,
    min_df=2,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluation
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
