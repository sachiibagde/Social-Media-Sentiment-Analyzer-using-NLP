import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


# Load data

df = pd.read_csv("twitter_training.csv", header=None)
df.columns = ["id", "game", "label", "text"]

texts = df["text"].astype(str)
labels = df["label"]

# Encode labels

le = LabelEncoder()
labels = le.fit_transform(labels)

# Tokenization

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=50)

# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Build LSTM model

model = Sequential([
    Embedding(10000, 128, input_length=50),
    LSTM(64),
    Dense(32, activation="relu"),
    Dense(len(set(labels)), activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Save

model.save("lstm_model.h5")

pickle.dump(tokenizer, open("lstm_tokenizer.pkl", "wb"))
pickle.dump(le, open("lstm_label_encoder.pkl", "wb"))

print("✅ LSTM model saved!")
