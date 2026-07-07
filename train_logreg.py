import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

print("🔥 train_logreg.py started")

# Load dataset (no header CSV)
train_df = pd.read_csv(
    "twitter_training.csv",
    header=None,
    names=["id", "entity", "sentiment", "text"]
)

print("📊 Columns:", train_df.columns)

# Extract data
texts = train_df["text"].astype(str)
labels = train_df["sentiment"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Vectorizer (same config as SVM)
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(texts)

# Train Logistic Regression
logreg_model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

logreg_model.fit(X, y)

# Save model
with open("logistic_regression_model.pkl", "wb") as f:
    pickle.dump(logreg_model, f)

print("✅ Logistic Regression trained and saved as logistic_regression_model.pkl")
