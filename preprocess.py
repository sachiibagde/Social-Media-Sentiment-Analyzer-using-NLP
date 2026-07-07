import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet',   quiet=True)

# ── Label definitions ─────────────────────────────────────────
LABEL_NAMES = ["Negative", "Neutral", "Positive"]

# ── Stopwords setup ───────────────────────────────────────────
_stop_words = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()

# ── Negation words — NEVER remove from text ──────────────────
# Removing "not" turns "not like" -> "like" -> wrong Positive
# Removing "does" turns "does not like" -> "not like" or "like"
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none",
    "nothing", "nobody", "nowhere", "hardly", "scarcely",
    "barely", "without", "cannot", "cant", "wont",
    "dont", "doesnt", "didnt", "isnt", "wasnt",
    "arent", "werent", "hasnt", "havent", "hadnt",
    "shouldnt", "wouldnt", "couldnt", "musnt",
    # Keep auxiliary verbs so "does not" / "do not" stays intact
    "does", "do", "did",
}

# Remove negation words from stopword set so they are preserved
_stop_words -= NEGATION_WORDS


def _attach_negation(tokens: list) -> list:
    """
    Merge negation + following word into single token.
    ["not", "like"]      -> ["not_like"]
    ["does", "not", "like"] -> ["does_not_like"]
    Preserves sentiment-flipping context as one feature.
    """
    result = []
    i = 0
    while i < len(tokens):
        # Handle "does/do/did + not + word" -> "does_not_word"
        if (tokens[i] in {"does", "do", "did"}
                and i + 2 < len(tokens)
                and tokens[i + 1] == "not"):
            result.append(f"{tokens[i]}_not_{tokens[i + 2]}")
            i += 3
        # Handle simple "not/never/no + word" -> "not_word"
        elif tokens[i] in NEGATION_WORDS and i + 1 < len(tokens):
            result.append(f"{tokens[i]}_{tokens[i + 1]}")
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


def clean_text(text: str) -> str:
    """
    Clean and normalize input text for sentiment analysis.
    Steps:
      1. Lowercase
      2. Normalize contractions  (can't->cant, won't->wont, does not->doesnt)
      3. Remove URLs, @mentions, #hashtags
      4. Keep alphanumeric only  (numbers kept for time/date context)
      5. Tokenize
      6. Attach negations        (does not like -> does_not_like)
      7. Remove stopwords        (negation words preserved)
      8. Lemmatize
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Normalize multi-word negations FIRST (before removing spaces/punct)
    text = re.sub(r"does\s+not\b", "doesnt", text)
    text = re.sub(r"do\s+not\b",   "dont",   text)
    text = re.sub(r"did\s+not\b",  "didnt",  text)
    text = re.sub(r"is\s+not\b",   "isnt",   text)
    text = re.sub(r"was\s+not\b",  "wasnt",  text)
    text = re.sub(r"are\s+not\b",  "arent",  text)
    text = re.sub(r"were\s+not\b", "werent", text)
    text = re.sub(r"has\s+not\b",  "hasnt",  text)
    text = re.sub(r"have\s+not\b", "havent", text)
    text = re.sub(r"had\s+not\b",  "hadnt",  text)
    text = re.sub(r"can\s+not\b",  "cant",   text)
    text = re.sub(r"will\s+not\b", "wont",   text)

    # Normalize apostrophe contractions
    contractions = {
        "can't": "cant",     "won't": "wont",      "don't": "dont",
        "doesn't": "doesnt", "didn't": "didnt",    "isn't": "isnt",
        "wasn't": "wasnt",   "aren't": "arent",    "weren't": "werent",
        "hasn't": "hasnt",   "haven't": "havent",  "hadn't": "hadnt",
        "shouldn't": "shouldnt", "wouldn't": "wouldnt",
        "couldn't": "couldnt",   "mustn't": "musnt",
        "it's": "its",       "i'm": "im",           "i've": "ive",
        "i'll": "ill",       "i'd": "id",
    }
    for c, e in contractions.items():
        text = text.replace(c, e)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove @mentions and #hashtag symbols
    text = re.sub(r'\@\w+|\#', '', text)

    # Keep letters and digits (numbers help with time/date context)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Attach negations before stopword removal
    tokens = _attach_negation(tokens)

    # Remove stopwords; always keep negation combo tokens (contain '_')
    tokens = [
        _lemmatizer.lemmatize(w)
        for w in tokens
        if (w not in _stop_words and len(w) > 2) or '_' in w
    ]

    return ' '.join(tokens)