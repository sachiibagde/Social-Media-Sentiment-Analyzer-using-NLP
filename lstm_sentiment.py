import pickle
import numpy as np
import os

# ── Safe model loading (won't crash Flask if files are missing) 
model = None
tokenizer = None
le = None

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    if not os.path.exists("lstm_model.h5"):
        raise FileNotFoundError("lstm_model.h5 not found")
    if not os.path.exists("lstm_tokenizer.pkl"):
        raise FileNotFoundError("lstm_tokenizer.pkl not found")
    if not os.path.exists("lstm_label_encoder.pkl"):
        raise FileNotFoundError("lstm_label_encoder.pkl not found")

    model = load_model("lstm_model.h5")
    tokenizer = pickle.load(open("lstm_tokenizer.pkl", "rb"))
    le = pickle.load(open("lstm_label_encoder.pkl", "rb"))
    print("✓ LSTM model loaded successfully.")

except Exception as e:
    print(f"⚠️  LSTM model not loaded (non-fatal): {e}")
    print("   App will continue using XLM model only.")


def predict(text):
    """
    Predict sentiment using LSTM model.
    Returns a fallback dict if model files are missing.
    """
    if model is None or tokenizer is None or le is None:
        return {
            "label": "Unavailable",
            "confidence": 0.0,
            "note": "LSTM model files not found. Train the model first."
        }

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=50)

    pred = model.predict(padded, verbose=0)[0]

    label = le.inverse_transform([np.argmax(pred)])[0]
    confidence = float(np.max(pred))

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }
