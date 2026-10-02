# scripts/measure_rag_threshold.py
"""
Prints the best cosine similarity each probe question gets against the RAG
knowledge base, so you can pick `rag.min_similarity` in configs/config.yaml.

A good threshold sits between the lowest in-scope score and the highest
off-topic score. General PakWheels questions without a KB section (e.g.
"how do I write a review") can fall either side; that's fine, because the
fine-tuned model answers them without context.

Run from the repo root:  python scripts/measure_rag_threshold.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.rag import RAGEngine  # noqa: E402
from src.utils import get_config  # noqa: E402

IN_SCOPE = [
    "How much does a featured ad cost?",
    "Sell It For Me charges for an SUV",
    "Is car inspection available in Sialkot?",
    "inspection ki fee kitni hai",
    "How do I verify an auction sheet?",
    "What is your helpline number?",
    "Can I return car mats I ordered?",
    "Do you offer car insurance?",
    "My ad was rejected, why?",
    "How do I reset my password?",
]
OFF_TOPIC = [
    "What's the weather in Lahore?",
    "Write me a python script",
    "Who won the cricket match?",
    "Give me a biryani recipe",
    "What is the capital of France?",
]


def best_score(store, question):
    results = store.similarity_search_with_score(question, k=1)
    return (1 - results[0][1], results[0][0].metadata.get("title")) if results else (0.0, None)


def main():
    config = get_config()
    rag = RAGEngine()
    docs = []
    for path in config["rag"]["documents"]:
        docs += rag.load_documents(path)
    store = rag.build_index(docs)

    print(f"\nCurrent rag.min_similarity = {config['rag']['min_similarity']}\n")
    scores = {}
    for label, questions in (("IN-SCOPE", IN_SCOPE), ("OFF-TOPIC", OFF_TOPIC)):
        print(label)
        scores[label] = []
        for q in questions:
            score, title = best_score(store, q)
            scores[label].append(score)
            print(f"  {score:.3f}  {q!r:50} -> {title}")
        print()

    low_in, high_off = min(scores["IN-SCOPE"]), max(scores["OFF-TOPIC"])
    print(f"Lowest in-scope: {low_in:.3f} | highest off-topic: {high_off:.3f}")
    if low_in > high_off:
        print(f"Suggested min_similarity: {(low_in + high_off) / 2:.2f}")
    else:
        print("Scores overlap: pick a value that keeps the fee/price questions above it.")


if __name__ == "__main__":
    main()
