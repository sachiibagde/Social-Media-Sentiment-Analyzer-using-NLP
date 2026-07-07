import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os, json, pickle, re, traceback, platform
from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator

from preprocess      import clean_text, LABEL_NAMES
from ensemble_models import EnsembleModels
from explain         import confidence_explanation

# XLM and LSTM — lazy import to avoid torch DLL crash on Windows
XLMSentiment   = None
lstm_sentiment = None

def _lazy_load_deep_models():
    global XLMSentiment, lstm_sentiment
    try:
        from xlm_sentiment import XLMSentiment as _XLM
        XLMSentiment = _XLM
    except Exception as e:
        print(f"⚠️  Could not import XLMSentiment: {e}")
    try:
        import lstm_sentiment as _lstm
        lstm_sentiment = _lstm
    except Exception as e:
        print(f"⚠️  Could not import lstm_sentiment: {e}")

_lazy_load_deep_models()

app = Flask(__name__)

COLOR_MAP = {
    "Positive": "#00C851",
    "Negative": "#ff4444",
    "Neutral":  "#ffbb33",
}

# STRONG SENTIMENT WORD LISTS

STRONG_NEGATIVE_WORDS = {
    "dislike", "hate", "hated", "hating", "horrible", "terrible",
    "awful", "disgusting", "pathetic", "useless", "worthless",
    "worst", "garbage", "trash", "scam", "fraud", "broken",
    "disappointing", "disappointed", "disappointment", "frustrating",
    "frustrated", "frustration", "annoying", "annoyed",
    "angry", "anger", "furious", "upset", "unhappy", "dissatisfied",
    "dissatisfaction", "complaint", "complain", "complained",
    "refund", "cancel", "cancelled", "failed", "failure",
    "unacceptable", "unprofessional", "rude", "dreadful",
    "appalling", "atrocious", "waste", "wasted", "regret",
    "boring", "bored", "dull", "ugly", "overpriced",
    "defective", "damaged", "delayed",
    # DO NOT add: bad, poor, worse, good, like — negation handles these
}

STRONG_POSITIVE_WORDS = {
    "love", "loved", "loving", "amazing", "excellent", "outstanding",
    "fantastic", "wonderful", "brilliant", "superb", "perfect",
    "awesome", "incredible", "extraordinary",
    "impressed", "impressive", "satisfied", "satisfaction",
    "happiness", "delighted", "delight", "thrilled",
    "recommended", "beautiful", "gorgeous", "stunning",
    "fabulous", "marvelous",
    # DO NOT add: great, good, like, best, happy — negation handles these
}


def _strong_word_override(text: str, label: str, conf: float):
    """
    If a strong sentiment word is found directly in the raw text
    and current prediction is Neutral or wrong polarity, override it.
    This catches cases like 'i dislike it' where the model is uncertain.
    """
    lower = text.lower()
    words = set(re.findall(r'\b\w+\b', lower))

    has_neg = bool(words & STRONG_NEGATIVE_WORDS)
    has_pos = bool(words & STRONG_POSITIVE_WORDS)

    # Check multi-word phrases too
    for phrase in ["never again", "do not recommend", "dont recommend",
                   "not recommend", "not satisfied", "not happy",
                   "no good", "highly recommend", "must buy",
                   "top notch", "world class", "five star", "5 star"]:
        if phrase in lower:
            if phrase in ["highly recommend", "must buy",
                          "top notch", "world class", "five star", "5 star"]:
                has_pos = True
            else:
                has_neg = True

    # Only override if BOTH positive and negative words present = mixed
    # skip override (ambiguous), otherwise apply
    if has_neg and not has_pos:
        if label in ("Neutral", "Positive"):
            print(f"  [strong-word] '{text}' -> overriding {label} -> Negative")
            return "Negative", max(conf, 0.65)

    if has_pos and not has_neg:
        if label in ("Neutral", "Negative"):
            print(f"  [strong-word] '{text}' -> overriding {label} -> Positive")
            return "Positive", max(conf, 0.65)

    return label, conf




_NEG_POSITIVE_WORDS = (
    r"bad|terrible|awful|horrible|worst|poor|dreadful|"
    r"hate|wrong|fail|ugly|useless|boring|disappoint|"
    r"trash|garbage|waste|broken|faulty|annoying|frustrat|"
    r"pathetic|disgusting|unacceptable"
)
_NEG_NEGATIVE_WORDS = (
    r"good|great|amazing|excellent|wonderful|best|"
    r"like|love|enjoy|satisf|recommend|impress|"
    r"fantastic|awesome|brilliant|outstanding|superb|"
    r"pleased|glad|worth|helpful|useful|nice|fine|"
    r"happy|perfect|work|deliver|arrive|appreciate"
)
_NEGATION_TRIGGERS = (
    r"not|never|no|hardly|barely|scarcely|"
    r"does\s+not|do\s+not|did\s+not|"
    r"don't|doesn't|didn't|can't|won't|isn't|wasn't|aren't|weren't|"
    r"dont|doesnt|didnt|cant|wont|"
    r"cannot|isnt|wasnt|arent|werent"
)

_NEG_TO_POS = re.compile(
    r"\b(" + _NEGATION_TRIGGERS + r")\b.{0,40}\b(" + _NEG_POSITIVE_WORDS + r")\w*\b",
    re.IGNORECASE
)
_NEG_TO_NEG = re.compile(
    r"\b(" + _NEGATION_TRIGGERS + r")\b.{0,40}\b(" + _NEG_NEGATIVE_WORDS + r")\w*\b",
    re.IGNORECASE
)

def _apply_negation_override(text: str, label: str, conf: float):
    """
    'not bad'        -> Negative -> Positive
    'does not like'  -> Positive -> Negative
    """
    if _NEG_TO_POS.search(text):
        if label in ("Negative", "Neutral"):
            print(f"  [negation] flipped {label} -> Positive  |  '{text}'")
            return "Positive", max(conf, 0.65)
    if _NEG_TO_NEG.search(text):
        if label in ("Positive", "Neutral"):
            print(f"  [negation] flipped {label} -> Negative  |  '{text}'")
            return "Negative", max(conf, 0.65)
    return label, conf



# LANGUAGE DETECTION

def _is_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    return sum(1 for c in letters if ord(c) < 128) / len(letters) >= 0.70


def _is_hinglish(text: str) -> bool:
    MARKERS = {
        "nhi","nahi","hai","hain","mera","meri","mere","tera","teri",
        "uska","unka","kya","koi","kuch","aaya","aaye","gaya","gaye",
        "hua","hui","hue","bahut","bohat","accha","acha","bura","buri",
        "wala","wali","abhi","tak","lekin","kyun","kyunki","isliye",
        "phir","bhi","sirf","bilkul","zaroor","shukriya","theek","thik",
        "dikkat","paisa","paise","baar","yaar","bhai","dost",
    }
    return len(set(text.lower().split()) & MARKERS) >= 2



# STARTUP

print("=" * 65)
print("  SENTIMENT ANALYSIS — STARTING UP")
print("=" * 65)

xlm_model = None
try:
    if XLMSentiment is None:
        raise ValueError("XLMSentiment import failed")
    xlm_model = XLMSentiment()
    print("✓ XLM model loaded.")
except Exception as e:
    print(f"❌ XLM failed to load: {e}")

ensemble    = EnsembleModels()
ensemble_ok = ensemble.load_models()
if not ensemble_ok:
    ensemble = None

best_model_name = "RF"
try:
    if os.path.exists("best_model_name.pkl"):
        with open("best_model_name.pkl", "rb") as f:
            best_model_name = pickle.load(f)
        print(f"✓ Best model key: {best_model_name}")
    else:
        print("⚠️  best_model_name.pkl missing — defaulting to RF")
except Exception as e:
    print(f"⚠️  Could not load best_model_name.pkl: {e}")

print("=" * 65)


# HELPERS


def translate_to_english(text: str) -> str:
    try:
        result = GoogleTranslator(source="auto", target="en").translate(text)
        if not result or not isinstance(result, str):
            return text
        return result
    except Exception:
        return text


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load {path}: {e}")
        return {}

# ROUTES

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test')
def test():
    return "FLASK OK"

@app.route('/ping')
def ping():
    return "FLASK ALIVE"


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # ── Parse input 
        if request.is_json:
            data = request.get_json(silent=True) or {}
            text = data.get("text", "")
        else:
            text = request.form.get("text", "")

        text = text.strip()
        if not text:
            return jsonify({"error": "Empty text"}), 400

        raw_text = text   # keep for overrides + language detection

        # ── Language detection ────────────────────────────────
        is_english  = _is_english(raw_text)
        is_hinglish = _is_hinglish(raw_text)
        is_foreign  = not is_english and not is_hinglish
        print(f"  [lang] english={is_english} hinglish={is_hinglish} foreign={is_foreign}")

        # ── Translate to English for RF/LSTM 
        original_text = text
        try:
            translated = translate_to_english(text)
            if not translated or not isinstance(translated, str) or not translated.strip():
                translated = original_text
        except Exception as e:
            print(f"⚠️  Translation skipped: {e}")
            translated = original_text

        text = translated
        print(f"  [translate] '{raw_text[:50]}' -> '{text[:50]}'")

        model_results = {}

        # ── XLM — raw text (natively multilingual) 
        try:
            if xlm_model is None:
                raise ValueError("XLM not loaded")
            r = xlm_model.predict(raw_text)
            model_results["XLM"] = {
                "label":      r["label"],
                "confidence": float(r["confidence"]),
                "all_probs":  r.get("all_probs", {})
            }
        except Exception as e:
            print(f"❌ XLM predict: {e}")
            model_results["XLM"] = {"label": "Error", "confidence": 0.0}

        # ── LSTM — translated English 
        try:
            if lstm_sentiment is None:
                raise ValueError("lstm_sentiment not loaded")
            r = lstm_sentiment.predict(text)
            model_results["LSTM"] = {
                "label":      r["label"],
                "confidence": float(r["confidence"]),
                "all_probs":  r.get("all_probs", {})
            }
        except Exception as e:
            print(f"❌ LSTM predict: {e}")
            model_results["LSTM"] = {"label": "Error", "confidence": 0.0}

        # ── Ensemble + individual ML models 
        try:
            if ensemble is None:
                raise ValueError("Ensemble not loaded")
            label, conf, probs_arr = ensemble.predict(text)
            flat = probs_arr[0]
            probs_dict = {
                "Negative": round(float(flat[0]), 4),
                "Neutral":  round(float(flat[1]), 4),
                "Positive": round(float(flat[2]), 4),
            }
            for key in ("ML", "RF", "LR", "SVM"):
                model_results[key] = {
                    "label":      label,
                    "confidence": round(conf, 4),
                    "all_probs":  probs_dict,
                }
            # Use individual model if best is RF/LR/SVM
            if best_model_name in ("RF", "LR", "SVM"):
                ind_label, ind_conf = ensemble.predict_single(text, best_model_name)
                if ind_label not in (None, "Error"):
                    model_results[best_model_name] = {
                        "label":      ind_label,
                        "confidence": round(ind_conf, 4),
                        "all_probs":  probs_dict,
                    }
        except Exception as e:
            print(f"❌ Ensemble predict: {e}")
            for key in ("ML", "RF", "LR", "SVM"):
                model_results[key] = {"label": "Error", "confidence": 0.0}

        # ── Smart routing 
        if is_foreign:
            xlm_res = model_results.get("XLM", {})
            if xlm_res.get("label") not in (None, "Error"):
                final      = xlm_res
                used_model = "XLM (multilingual)"
            else:
                final      = model_results.get(best_model_name, {})
                used_model = best_model_name + " (XLM fallback)"
        else:
            final      = model_results.get(best_model_name, {})
            used_model = best_model_name + (" [Hinglish→EN]" if is_hinglish else "")

        # Fallback if chosen model errored
        if not final or final.get("label") in (None, "Error", "Unavailable"):
            for key in ("RF", "XLM", "LR", "SVM", "LSTM", "ML"):
                candidate = model_results.get(key, {})
                if candidate.get("label") not in (None, "Error", "Unavailable"):
                    final      = candidate
                    used_model = key + " (fallback)"
                    break
            else:
                return jsonify({"error": "All models failed"}), 500

        final_label = final["label"]
        final_conf  = float(final["confidence"])

        # ── Override pipeline (applied on raw text) ───────────

        # Step 1: Strong sentiment word override
        # "dislike/hate/love/amazing" -> correct Neutral/wrong predictions
        # Applied FIRST so negation can override it if needed
        final_label, final_conf = _strong_word_override(
            raw_text, final_label, final_conf
        )

        # Step 2: Negation override — ALWAYS applied last so it wins
        # "not bad" -> Positive | "does not like" -> Negative
        # Negation MUST come after strong-word so "not bad" flips back
        final_label, final_conf = _apply_negation_override(
            raw_text, final_label, final_conf
        )

        # Step 3: Low-confidence fallback -> Neutral
        # Only apply if NO strong words found (handled above)
        LOW_CONF_THRESHOLD = 0.45
        lower_raw = raw_text.lower()
        raw_words = set(re.findall(r'\b\w+\b', lower_raw))
        has_strong = bool(
            (raw_words & STRONG_NEGATIVE_WORDS) or
            (raw_words & STRONG_POSITIVE_WORDS)
        )
        if final_conf < LOW_CONF_THRESHOLD and not has_strong:
            final_label = "Neutral"

        # ── Important words 
        important_words = []
        try:
            if xlm_model:
                important_words = xlm_model.get_important_words(raw_text)
        except Exception:
            pass

        explanation = confidence_explanation(final_conf)

        return jsonify({
            "sentiment":        final_label,
            "confidence":       f"{final_conf * 100:.1f}%",
            "confidence_value": round(final_conf, 4),
            "color":            COLOR_MAP.get(final_label, "#ffbb33"),
            "explanation":      explanation,
            "important_words":  important_words,
            "feedback": [
                f"🏆 Prediction from: {used_model}",
                f"Detected sentiment: {final_label.lower()}",
                f"{'🌐 Multilingual mode' if is_foreign else '🌍 Hinglish mode' if is_hinglish else '🇬🇧 English mode'}"
            ],
            "best_model": {
                "name":       used_model,
                "confidence": final_conf,
            },
            "all_models": model_results,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    data = _load_json("confusion_results.json")
    if not data:
        return jsonify({"error": "confusion_results.json not found."}), 404
    summary = []
    for model_name, metrics in data.items():
        summary.append({
            "model":            model_name,
            "accuracy":         metrics.get("accuracy",  0),
            "precision":        metrics.get("precision", 0),
            "recall":           metrics.get("recall",    0),
            "f1_score":         metrics.get("f1_score",  metrics.get("f1", 0)),
            "confusion_matrix": metrics.get("confusion_matrix", metrics.get("matrix", [])),
            "labels":           metrics.get("labels", LABEL_NAMES),
            "per_class":        metrics.get("per_class", {}),
        })
    return jsonify({"models": summary, "classes": LABEL_NAMES})


@app.route("/api/roc", methods=["GET"])
def api_roc():
    data = _load_json("roc_results.json")
    if not data:
        return jsonify({"error": "roc_results.json not found."}), 404
    return jsonify(data)


@app.route("/api/trend", methods=["GET"])
def api_trend():
    data = _load_json("confusion_results.json")
    if not data:
        return jsonify({"error": "confusion_results.json not found."}), 404
    first     = next(iter(data.values()), {})
    per_class = first.get("per_class", {})
    trend     = {cls: per_class.get(cls, {}).get("support", 0) for cls in LABEL_NAMES}
    return jsonify({"class_distribution": trend, "labels": LABEL_NAMES})



# ENTRY POINT

if __name__ == '__main__':
    print("\n🚀 Flask server starting...")
    print("📡 http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    is_windows = platform.system() == 'Windows'
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=not is_windows,
        threaded=True
    )