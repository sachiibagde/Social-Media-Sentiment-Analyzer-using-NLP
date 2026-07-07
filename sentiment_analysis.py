import pandas as pd
import numpy as np
import pickle
import os
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

print("=" * 65)
print("  SENTIMENT ANALYSIS — TRAINING SCRIPT")
print("=" * 65)


# ── 1. LOAD DATA 
print("\n[1/7] Loading dataset...")

TRAIN_PATH = "twitter_training.csv"
VAL_PATH   = "twitter_validation.csv"

train_df = pd.read_csv(TRAIN_PATH, header=None,
                       names=['id', 'topic', 'sentiment', 'text'])
val_df   = pd.read_csv(VAL_PATH,   header=None,
                       names=['id', 'topic', 'sentiment', 'text'])

print(f"  Train: {len(train_df):,} rows")
print(f"  Val:   {len(val_df):,} rows")
print(f"  Train classes: {train_df['sentiment'].value_counts().to_dict()}")


# ── 2. REMOVE IRRELEVANT + CLEAN 
print("\n[2/7] Cleaning data (removing Irrelevant class)...")

# Drop Irrelevant
train_df = train_df[train_df['sentiment'] != 'Irrelevant'].copy()
val_df   = val_df[val_df['sentiment']     != 'Irrelevant'].copy()

# Drop nulls
train_df.dropna(subset=['text'], inplace=True)
val_df.dropna(subset=['text'],   inplace=True)

# Drop empty strings
train_df = train_df[train_df['text'].str.strip() != '']
val_df   = val_df[val_df['text'].str.strip()     != '']

# Reset index
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True,   inplace=True)

print(f"  Train after cleaning: {len(train_df):,} rows")
print(f"  Val after cleaning:   {len(val_df):,} rows")
print(f"  Classes: {train_df['sentiment'].value_counts().to_dict()}")


# ── 3. TEXT PREPROCESSING 
print("\n[3/7] Preprocessing text...")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)


train_df['clean_text'] = train_df['text'].apply(clean_text)
val_df['clean_text']   = val_df['text'].apply(clean_text)

# Drop rows where cleaning produced empty string
train_df = train_df[train_df['clean_text'] != ''].reset_index(drop=True)
val_df   = val_df[val_df['clean_text']     != ''].reset_index(drop=True)

print(f"  Done. Train: {len(train_df):,}  Val: {len(val_df):,}")


# ── 4. ENCODE LABELS 
print("\n[4/7] Encoding labels...")

LABELS = ['Negative', 'Neutral', 'Positive']

le = LabelEncoder()
le.fit(LABELS)
print("Label classes:", le.classes_)

y_train = le.transform(train_df['sentiment'])
y_val   = le.transform(val_df['sentiment'])

print(f"  Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Save label encoder for LSTM
with open('lstm_label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("  ✓ Saved lstm_label_encoder.pkl")


# ── 5. VECTORIZE (TF-IDF for ML models) 
print("\n[5/7] Vectorizing text (TF-IDF)...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_df['clean_text'])
X_val   = vectorizer.transform(val_df['clean_text'])

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"  ✓ Saved vectorizer.pkl  (vocab size: {len(vectorizer.vocabulary_):,})")


# ── 6. TRAIN ML MODELS 
print("\n[6/7] Training ML models...")

results = {}


def evaluate(name, model, X_v, y_v):
    preds = model.predict(X_v)
    acc  = accuracy_score(y_v, preds)
    prec = precision_score(y_v, preds, average='weighted', zero_division=0)
    rec  = recall_score(y_v, preds, average='weighted', zero_division=0)
    f1   = f1_score(y_v, preds, average='weighted', zero_division=0)
    results[name] = {
        'Accuracy':  round(acc  * 100, 1),
        'Precision': round(prec * 100, 1),
        'Recall':    round(rec  * 100, 1),
        'F1-Score':  round(f1   * 100, 1)
    }
    print(f"\n  [{name}]")
    print(f"    Accuracy:  {acc*100:.1f}%")
    print(f"    Precision: {prec*100:.1f}%")
    print(f"    Recall:    {rec*100:.1f}%")
    print(f"    F1-Score:  {f1*100:.1f}%")
    print(classification_report(y_v, preds,
                                target_names=le.classes_,
                                zero_division=0))
    return acc


# Logistic Regression
print("  Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, C=1.0,
                        solver='lbfgs', multi_class='multinomial',
                        n_jobs=-1, random_state=42)
lr.fit(X_train, y_train)
lr_acc = evaluate("Logistic Regression", lr, X_val, y_val)
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
print("  ✓ Saved logistic_regression_model.pkl")

# SVM
print("  Training SVM...")
svm = SVC(kernel='linear', probability=True, C=1.0,
          random_state=42, max_iter=2000)
svm.fit(X_train, y_train)
svm_acc = evaluate("SVM", svm, X_val, y_val)
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)
print("  ✓ Saved svm_model.pkl")

# Random Forest
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                             n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_acc = evaluate("Random Forest", rf, X_val, y_val)
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("  ✓ Saved random_forest_model.pkl")


# ── LSTM 
print("\n  Training LSTM...")
lstm_acc = 0.0

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Embedding, LSTM, Dense,
                                         Dropout, Bidirectional)
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping

    MAX_WORDS = 30000
    MAX_LEN   = 50
    EMBED_DIM = 128
    EPOCHS    = 10
    BATCH     = 256

    # Tokenize
    tok = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tok.fit_on_texts(train_df['clean_text'])

    X_train_seq = pad_sequences(
        tok.texts_to_sequences(train_df['clean_text']),
        maxlen=MAX_LEN, padding='post', truncating='post'
    )
    X_val_seq = pad_sequences(
        tok.texts_to_sequences(val_df['clean_text']),
        maxlen=MAX_LEN, padding='post', truncating='post'
    )

    # One-hot encode labels for Keras
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes=3)

    # Build model
    model = Sequential([
        Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    es = EarlyStopping(monitor='val_accuracy', patience=3,
                       restore_best_weights=True)

    history = model.fit(
        X_train_seq, y_train_cat,
        validation_data=(X_val_seq, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[es],
        verbose=1
    )

    # Evaluate
    y_pred_probs = model.predict(X_val_seq, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    lstm_acc     = accuracy_score(y_val, y_pred)

    prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_val, y_pred,    average='weighted', zero_division=0)
    f1   = f1_score(y_val, y_pred,        average='weighted', zero_division=0)

    results['LSTM'] = {
        'Accuracy':  round(lstm_acc * 100, 1),
        'Precision': round(prec     * 100, 1),
        'Recall':    round(rec      * 100, 1),
        'F1-Score':  round(f1       * 100, 1)
    }

    print(f"\n  [LSTM]")
    print(f"    Accuracy:  {lstm_acc*100:.1f}%")
    print(classification_report(y_val, y_pred,
                                target_names=le.classes_,
                                zero_division=0))

    # Save
    model.save('lstm_model.h5')
    with open('lstm_tokenizer.pkl', 'wb') as f:
        pickle.dump(tok, f)
    print("  ✓ Saved lstm_model.h5 and lstm_tokenizer.pkl")

except Exception as e:
    print(f"  ❌ LSTM training failed: {e}")
    results['LSTM'] = {
        'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1-Score': 0
    }


# ── 7. SELECT BEST MODEL 
print("\n[7/7] Selecting best model...")

model_acc = {
    'Logistic Regression': lr_acc,
    'SVM':                 svm_acc,
    'Random Forest':       rf_acc,
    'LSTM':                lstm_acc
}


flask_key_map = {
    'Logistic Regression': 'LR',
    'SVM':                 'SVM',
    'Random Forest':       'RF',
    'LSTM':                'LSTM'
}

best_name     = max(model_acc, key=model_acc.get)
best_acc      = model_acc[best_name]
best_flask_key = flask_key_map[best_name]

print(f"\n  🏆 Best model: {best_name} ({best_acc*100:.1f}%)")
print(f"     Flask key:  {best_flask_key}")

with open('best_model_name.pkl', 'wb') as f:
    pickle.dump(best_flask_key, f)
print("  ✓ Saved best_model_name.pkl")


# ── PLOT MODEL COMPARISON 
print("\n  Generating model comparison chart...")

model_names = list(results.keys())
metrics     = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors      = ['#6699ff', '#cc99ff', '#66cc66', '#ffcc44']

x     = np.arange(len(model_names))
width = 0.18

fig, ax = plt.subplots(figsize=(14, 6))

for i, (metric, color) in enumerate(zip(metrics, colors)):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(x + i * width, vals, width, label=metric,
                  color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=8)

ax.set_title('Model Performance Comparison (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Score (%)')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 110)
ax.legend()
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.close()
print("  ✓ Saved model_comparison.png")


# ── SUMMARY ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  TRAINING COMPLETE — FILES SAVED:")
print("=" * 65)
saved = [
    'vectorizer.pkl',
    'lstm_label_encoder.pkl',
    'logistic_regression_model.pkl',
    'svm_model.pkl',
    'random_forest_model.pkl',
    'lstm_model.h5',
    'lstm_tokenizer.pkl',
    'best_model_name.pkl',
    'model_comparison.png'
]
for f in saved:
    status = "✓" if os.path.exists(f) else "✗ MISSING"
    print(f"  {status}  {f}")

print(f"\n  🏆 Best model: {best_name} ({best_acc*100:.1f}% accuracy)")
print("\n  Now run:  python flask_app.py")
print("=" * 65)