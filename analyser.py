from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from custom_analyser import custom_score

# Load models
print("Loading models...")
sum_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
sentiment_model = pipeline("text-classification", model="ProsusAI/finbert")
print("Models ready.\n")


def summarise(text):
    """Summarise text using DistilBART."""
    inputs = sum_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    ids = sum_model.generate(inputs["input_ids"], max_length=80, min_length=20, do_sample=False)
    return sum_tokenizer.decode(ids[0], skip_special_tokens=True)


FINBERT_NUMERIC = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def combined_rating(avg):
    """Map a -1 to +1 average score to a 1-5 rating."""
    if avg < -0.5:
        return 1
    if avg < -0.15:
        return 2
    if avg < 0.15:
        return 3
    if avg < 0.5:
        return 4
    return 5


def combined_label(avg):
    """Map a -1 to +1 average score to a sentiment label."""
    if avg < -0.15:
        return "negative"
    if avg > 0.15:
        return "positive"
    return "neutral"


def analyse(text):
    """Summarise text, run FinBERT + custom analyser, return combined results."""
    word_count = len(text.split())
    summary = summarise(text) if word_count >= 30 else text

    # Chunk long texts so FinBERT's 512-token limit is never exceeded.
    # Each chunk is scored independently and the results are averaged.
    tokens = sentiment_model.tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunk_size = 512
    stride = 50  # overlap between chunks to avoid cutting mid-sentence
    chunks = []
    for i in range(0, len(tokens), chunk_size - stride):
        chunk = tokens[i: i + chunk_size]
        chunks.append(sentiment_model.tokenizer.decode(chunk, skip_special_tokens=True))
        if i + chunk_size >= len(tokens):
            break

    chunk_scores = []
    for chunk in chunks:
        r = sentiment_model(chunk, truncation=True, max_length=512)[0]
        chunk_scores.append(FINBERT_NUMERIC[r["label"]])

    finbert_num = sum(chunk_scores) / len(chunk_scores)
    finbert_label = "positive" if finbert_num > 0.15 else "negative" if finbert_num < -0.15 else "neutral"

    lex_score = custom_score(text)

    avg = finbert_num * 0.3 + lex_score * 0.7
    label = combined_label(avg)
    rating = combined_rating(avg)

    return summary, finbert_label, lex_score, avg, label, rating


def main():
    print("=== Financial Sentiment Analyser ===")
    print("Paste a financial news sentence or review, then press Enter.")
    print("Type 'quit' to exit.\n")

    while True:
        text = input("Input: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        print("\nAnalysing...")
        summary, finbert_label, lex_score, avg, label, rating = analyse(text)

        print(f"\n  Summary   : {summary}")
        print(f"  FinBERT   : {finbert_label.capitalize()}")
        print(f"  Custom    : {lex_score:+.2f}")
        print(f"  Combined  : {label.capitalize()} (avg {avg:+.2f})")
        print(f"  Rating    : {rating} / 5")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
