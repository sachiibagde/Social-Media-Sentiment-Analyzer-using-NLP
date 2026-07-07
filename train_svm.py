import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

print("🔥 train_svm.py started")

# Load dataset
train_df = pd.read_csv(
    "twitter_training.csv",
    header=None,
    names=["id", "entity", "sentiment", "text"]
)

print("📊 Columns:", train_df.columns)

# ⚠️ VERY IMPORTANT: column mapping
# Kaggle twitter dataset format:
# [id, entity, sentiment, text]

train_texts = train_df.iloc[:, 3].astype(str)
train_labels = train_df.iloc[:, 2]

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_labels)

# Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train = vectorizer.fit_transform(train_texts)

# Train SVM with probability
svm_model = SVC(
    kernel="linear",
    probability=True,
    random_state=42,
    max_iter=2000
)

svm_model.fit(X_train, y_train)

# Save model
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

print("✅ SVM model trained and saved as svm_model.pkl")
