def confidence_explanation(confidence: float) -> str:
    """Return a human-readable explanation for a confidence score."""
    if confidence >= 0.85:
        return "🔥 Strong sentiment detected with high confidence."
    elif confidence >= 0.60:
        return "🙂 Moderate sentiment detected from context."
    elif confidence >= 0.40:
        return "😐 Slight or mixed sentiment."
    else:
        return "🤔 Very weak sentiment or neutral text."