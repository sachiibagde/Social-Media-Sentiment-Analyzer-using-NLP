from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re


def split_sentences(text):
    """Split long paragraph into smaller sentences."""
    parts = re.split(r'[.!?।]', text)
    return [p.strip() for p in parts if len(p.strip()) > 5]


class XLMSentiment:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        self.labels = ["Negative", "Neutral", "Positive"]

    def _predict_single(self, text):
        """Predict sentiment for one sentence."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)[0]
        confidence, idx = torch.max(probs, dim=0)

        return {
            "label": self.labels[idx.item()],
            "confidence": confidence.item()
        }

    def get_important_words(self, text, top_k=3):
        """Get important words using attention scores."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        attentions = outputs.attentions[-1]
        scores = attentions.mean(dim=1).mean(dim=1)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        pairs = list(zip(tokens, scores.tolist()))
        pairs = [
            (t, s) for t, s in pairs
            if not t.startswith("<") and len(t) > 2
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)
        important = [t.replace("▁", "") for t, _ in pairs[:top_k]]

        return important

    def predict(self, text):
        """Main predict — handles paragraphs via sentence splitting."""
        sentences = split_sentences(text)

        if not sentences:
            sentences = [text]

        results = []
        for s in sentences:
            res = self._predict_single(s)
            results.append(res)

        # Majority voting
        labels = [r["label"] for r in results]
        final_label = max(set(labels), key=labels.count)

        # Average confidence
        confidence = sum(r["confidence"] for r in results) / len(results)

        # Negation correction
        text_lower = text.lower()
        if any(neg in text_lower for neg in ["not", "नहीं", "नाही"]):
            if final_label == "Positive":
                final_label = "Negative"

        return {
            "label": final_label,
            "confidence": round(confidence, 4),
            "sentence_count": len(sentences)
        }
