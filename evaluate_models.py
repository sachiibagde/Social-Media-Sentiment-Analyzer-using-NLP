import os, json, warnings, pickle
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import label_binarize
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     confusion_matrix, roc_curve, auc)
from tensorflow.keras.models                   import load_model
from tensorflow.keras.preprocessing.sequence  import pad_sequences
from xlm_sentiment import XLMSentiment


# CONFIGURATION

CSV_FILE   = "twitter_training.csv"
VECTORIZER = "vectorizer.pkl"
LR_MODEL   = "logistic_regression_model.pkl"
SVM_MODEL  = "svm_model.pkl"
RF_MODEL   = "random_forest_model.pkl"
LSTM_MODEL = "lstm_model.h5"
LSTM_TOK   = "lstm_tokenizer.pkl"

LSTM_MAXLEN  = 100   # ← change if your training used a different value
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# The canonical class order used everywhere in this script
CLASSES = ["Negative", "Neutral", "Positive"]   # alphabetical = 0, 1, 2
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


print("\n" + "=" * 60)
print("   SENTISENSE  —  Model Evaluation  (v3 root-cause fix)")
print("=" * 60 + "\n")


# 1. LOAD DATASET  — keep labels as STRINGS throughout

print("📂  Loading dataset ...")

df = pd.read_csv(CSV_FILE, header=None)
df.columns = ["id", "game", "label", "text"]
df = df.dropna(subset=["text", "label"])
df = df[df["label"].isin(CLASSES)]

X_all = df["text"].astype(str).values
y_str = df["label"].values          # <-- strings: "Positive" / "Negative" / "Neutral"
y_int = np.array([CLASS_TO_IDX[c] for c in y_str])  # integer version

X_train, X_test, y_train_str, y_test_str = train_test_split(
    X_all, y_str,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_int,
)
_, _, y_train_int, y_test_int = train_test_split(
    X_all, y_int,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_int,
)

print(f"   Train: {len(X_train):,}   Test: {len(X_test):,}")
print(f"   Class distribution in test set:")
for c in CLASSES:
    n = (y_test_str == c).sum()
    print(f"     {c}: {n} ({n/len(y_test_str)*100:.1f}%)")
print()


# 2. STORAGE

results   = {}   # name → [acc, pre, rec, f1]
cm_store  = {}   # name → ndarray
roc_store = {}   # name → {class: {fpr, tpr, auc}}



# 3. HELPER FUNCTIONS

def record(name, y_true_str, y_pred_str):
    """All inputs are STRING labels. Compute metrics against CLASSES order."""
    acc = accuracy_score (y_true_str, y_pred_str)
    pre = precision_score(y_true_str, y_pred_str, labels=CLASSES,
                          average="weighted", zero_division=0)
    rec = recall_score   (y_true_str, y_pred_str, labels=CLASSES,
                          average="weighted", zero_division=0)
    f1  = f1_score       (y_true_str, y_pred_str, labels=CLASSES,
                          average="weighted", zero_division=0)
    results[name]  = [acc, pre, rec, f1]
    cm_store[name] = confusion_matrix(y_true_str, y_pred_str, labels=CLASSES)
    print(f"   {name:<22}  acc={acc*100:.1f}%  pre={pre*100:.1f}%  "
          f"rec={rec*100:.1f}%  f1={f1*100:.1f}%")


def record_roc(name, y_true_str, y_score_2d):
    """y_score_2d columns must align with CLASSES order."""
    y_bin = label_binarize(
        [CLASS_TO_IDX[c] for c in y_true_str],
        classes=[0, 1, 2]
    )
    data = {}
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score_2d[:, i])
        auc_val = auc(fpr, tpr)
        data[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                     "auc": round(float(auc_val), 4)}
    roc_store[name] = data


def save_cm_image(name, cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ticks = range(len(CLASSES))
    ax.set_xticks(list(ticks)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(list(ticks)); ax.set_yticklabels(CLASSES)
    thresh = cm.max() / 2.0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    fname = f"{name.replace(' ', '_')}_confusion.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"   ✓ Saved {fname}")


def model_classes_to_strings(model):
    """
    Extract string class labels from a fitted sklearn model.
    Returns list like ['Negative','Neutral','Positive'] in the
    order the model uses internally (model.classes_).
    """
    if hasattr(model, "classes_"):
        return [str(c) for c in model.classes_]
    # Pipeline: check the final estimator
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                return [str(c) for c in step.classes_]
    return None


def align_proba(proba, model_classes):
    """
    Reorder probability columns so they match CLASSES order.
    model_classes = list of class labels in the model's internal order.
    """
    if model_classes is None or model_classes == CLASSES:
        return proba
    # Build index mapping: for each position in CLASSES, find column in proba
    col_map = []
    for cls in CLASSES:
        if cls in model_classes:
            col_map.append(model_classes.index(cls))
        else:
            col_map.append(None)
    aligned = np.zeros((proba.shape[0], len(CLASSES)))
    for out_idx, in_idx in enumerate(col_map):
        if in_idx is not None:
            aligned[:, out_idx] = proba[:, in_idx]
    return aligned


def sklearn_predict_strings(model, X):
    """
    Get string predictions from a sklearn model regardless of
    whether its internal classes_ are strings or integers.
    """
    raw_preds  = model.predict(X)
    model_cls  = model_classes_to_strings(model)

    # If model was trained with integer labels, map back via model_cls
    if model_cls and model_cls[0] in CLASSES:
        return np.array(raw_preds, dtype=str)

    # If model was trained with integer labels (0,1,2) — map using CLASSES
    try:
        return np.array([CLASSES[int(p)] for p in raw_preds])
    except (ValueError, IndexError):
        return np.array(raw_preds, dtype=str)


# 4. SKLEARN MODELS

vectorizer  = pickle.load(open(VECTORIZER, "rb"))
X_vec_test  = vectorizer.transform(X_test)


def debug_model_classes(name, model):
    cls = model_classes_to_strings(model)
    print(f"   [{name}] internal classes_ → {cls}")

# Logistic Regression 
print("🔹  Logistic Regression")
lr = pickle.load(open(LR_MODEL, "rb"))
debug_model_classes("LR", lr)

lr_pred_str = sklearn_predict_strings(lr, X_vec_test)
print(f"   Sample predictions: {lr_pred_str[:8]}")
print(f"   Sample ground truth: {y_test_str[:8]}")

record("Logistic Regression", y_test_str, lr_pred_str)
try:
    lr_proba = lr.predict_proba(X_vec_test)
    lr_cls   = model_classes_to_strings(lr)
    lr_proba = align_proba(lr_proba, lr_cls)
    record_roc("Logistic Regression", y_test_str, lr_proba)
except Exception as e:
    print(f"   ⚠ ROC skipped: {e}")

#  SVM 
print("\n🔹  SVM")
svm_m = pickle.load(open(SVM_MODEL, "rb"))
debug_model_classes("SVM", svm_m)

svm_pred_str = sklearn_predict_strings(svm_m, X_vec_test)
print(f"   Sample predictions: {svm_pred_str[:8]}")

record("SVM", y_test_str, svm_pred_str)
try:
    svm_proba = svm_m.predict_proba(X_vec_test)
    svm_cls   = model_classes_to_strings(svm_m)
    svm_proba = align_proba(svm_proba, svm_cls)
    record_roc("SVM", y_test_str, svm_proba)
except Exception:
    try:
        df_sc     = svm_m.decision_function(X_vec_test)
        exp_sc    = np.exp(df_sc - df_sc.max(axis=1, keepdims=True))
        svm_proba = exp_sc / exp_sc.sum(axis=1, keepdims=True)
        svm_cls   = model_classes_to_strings(svm_m)
        svm_proba = align_proba(svm_proba, svm_cls)
        record_roc("SVM", y_test_str, svm_proba)
    except Exception as e2:
        print(f"   ⚠ ROC skipped: {e2}")

#  Random Forest
print("\n🔹  Random Forest")
rf = pickle.load(open(RF_MODEL, "rb"))
debug_model_classes("RF", rf)

rf_pred_str = sklearn_predict_strings(rf, X_vec_test)
record("Random Forest", y_test_str, rf_pred_str)
try:
    rf_proba = rf.predict_proba(X_vec_test)
    rf_cls   = model_classes_to_strings(rf)
    rf_proba = align_proba(rf_proba, rf_cls)
    record_roc("Random Forest", y_test_str, rf_proba)
except Exception as e:
    print(f"   ⚠ ROC skipped: {e}")


# 5. LSTM  —  auto-detect maxlen + integer label mapping

print("\n🔹  LSTM")

lstm_model = load_model(LSTM_MODEL)
tokenizer  = pickle.load(open(LSTM_TOK, "rb"))

# Auto-detect maxlen from model input shape
try:
    detected_maxlen = lstm_model.input_shape[1]
    print(f"   Auto-detected maxlen from model: {detected_maxlen}")
    LSTM_MAXLEN = detected_maxlen
except Exception:
    print(f"   Using configured LSTM_MAXLEN = {LSTM_MAXLEN}")

seq    = tokenizer.texts_to_sequences(X_test)
padded = pad_sequences(seq, maxlen=LSTM_MAXLEN)

lstm_proba = lstm_model.predict(padded, batch_size=64, verbose=0)
lstm_pred_int = np.argmax(lstm_proba, axis=1)

# LSTM output shape tells us how many classes
n_lstm_classes = lstm_proba.shape[1]
print(f"   LSTM output classes: {n_lstm_classes}")
print(f"   Unique predictions : {np.unique(lstm_pred_int)}")


LSTM_LABEL_MAP = {
    0: "Negative",   # ← update if your training used different order
    1: "Neutral",
    2: "Positive",
}
print(f"   LSTM label map: {LSTM_LABEL_MAP}")

lstm_pred_str = np.array([LSTM_LABEL_MAP[i] for i in lstm_pred_int])
print(f"   Sample LSTM preds : {lstm_pred_str[:8]}")
print(f"   Sample ground truth: {y_test_str[:8]}")

# Align LSTM probability columns to CLASSES order
lstm_class_order = [LSTM_LABEL_MAP[i] for i in range(n_lstm_classes)]
lstm_proba_aligned = align_proba(lstm_proba, lstm_class_order)

record("LSTM", y_test_str, lstm_pred_str)
record_roc("LSTM", y_test_str, lstm_proba_aligned)


# 6. XLM-RoBERTa  —  full test set

print("\n🔹  XLM-RoBERTa  (full test set — takes a few minutes ...)")

xlm_model    = XLMSentiment()
xlm_pred_str = []
xlm_proba    = []

for i, text in enumerate(X_test):
    res     = xlm_model.predict(str(text))
    lbl     = res.get("label", "Neutral")
    if lbl not in CLASSES:
        lbl = "Neutral"
    xlm_pred_str.append(lbl)

    conf  = float(res.get("confidence", 0.70))
    probs = [(1.0 - conf) / 2.0] * 3
    probs[CLASS_TO_IDX[lbl]] = conf
    xlm_proba.append(probs)

    if (i + 1) % 200 == 0:
        print(f"   ... {i+1}/{len(X_test)}  ({(i+1)/len(X_test)*100:.0f}%)")

xlm_pred_str = np.array(xlm_pred_str)
xlm_proba    = np.array(xlm_proba)

record("XLM-RoBERTa", y_test_str, xlm_pred_str)
record_roc("XLM-RoBERTa", y_test_str, xlm_proba)


# 7. RESULTS TABLE

print("\n" + "=" * 60)
print(f"   {'Model':<22} {'Acc':>7} {'Pre':>7} {'Rec':>7} {'F1':>7}")
print("   " + "-" * 52)
for name, v in results.items():
    print(f"   {name:<22} {v[0]*100:>6.1f}% {v[1]*100:>6.1f}% "
          f"{v[2]*100:>6.1f}% {v[3]*100:>6.1f}%")
print("=" * 60 + "\n")


# 8. CONFUSION MATRIX IMAGES

print("📊  Saving confusion matrix images ...")
for name, cm in cm_store.items():
    save_cm_image(name, cm)


# 9. BAR CHART

print("\n📊  Saving comparison bar chart ...")

names_list  = list(results.keys())
metrics_pct = np.array([[v * 100 for v in results[n]] for n in names_list])

x      = np.arange(len(names_list))
width  = 0.18
colors = ["#4f7df9", "#a78bfa", "#22c55e", "#fbbf24"]
bar_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

fig, ax = plt.subplots(figsize=(13, 6))
for i, (col, lbl) in enumerate(zip(colors, bar_labels)):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, metrics_pct[:, i], width,
                  label=lbl, color=col, alpha=0.88,
                  edgecolor="white", linewidth=0.4)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(names_list, fontsize=11)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_ylim(0, 108)
ax.set_title("Model Performance Comparison (%)", fontsize=14, pad=14)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()
print("   ✓ Saved model_comparison.png")


# 10. JSON EXPORTS

print("\n💾  Exporting JSON files ...")

TAG_MAP = {
    "Logistic Regression": "Linear · Fast · TF-IDF",
    "SVM":                 "Max-margin · Robust",
    "Random Forest":       "Ensemble · High accuracy",
    "LSTM":                "Deep Learning · Sequential",
    "XLM-RoBERTa":        "Transformer · Multilingual",
}

def macro_auc_pct(name):
    if name not in roc_store:
        return 0.0
    vals = [v["auc"] for v in roc_store[name].values()]
    return round(float(np.mean(vals)) * 100, 1) if vals else 0.0

metrics_out = []
for name, vals in results.items():
    metrics_out.append({
        "model":     name,
        "accuracy":  round(vals[0] * 100, 1),
        "precision": round(vals[1] * 100, 1),
        "recall":    round(vals[2] * 100, 1),
        "f1":        round(vals[3] * 100, 1),
        "auc":       macro_auc_pct(name),
        "tags":      TAG_MAP.get(name, "ML Model"),
        "best":      False,
    })

best_idx = max(range(len(metrics_out)),
               key=lambda i: metrics_out[i]["accuracy"])
metrics_out[best_idx]["best"] = True

with open("eval_results.json", "w") as f:
    json.dump({"models": metrics_out}, f, indent=2)
print("   ✓ eval_results.json")

cm_out = {}
for name, cm in cm_store.items():
    cm_out[name] = {
        "labels": CLASSES,
        "matrix": cm.tolist(),
        "total":  int(cm.sum()),
    }
with open("confusion_results.json", "w") as f:
    json.dump(cm_out, f, indent=2)
print("   ✓ confusion_results.json")

roc_out = {}
for name, cls_data in roc_store.items():
    roc_out[name] = {
        "auc": {cls: round(v["auc"], 4) for cls, v in cls_data.items()}
    }
with open("roc_results.json", "w") as f:
    json.dump(roc_out, f, indent=2)
print("   ✓ roc_results.json")


print("\n" + "=" * 60)
print("   ✅  COMPLETE")
print("=" * 60)
