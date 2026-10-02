# scripts/evaluate.py

import os
import sys
from datasets import Dataset
from dotenv import find_dotenv, load_dotenv
from ragas import evaluate

from openai import OpenAI
from ragas.llms import llm_factory

from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
)

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.facts import build_context, guard
from src.rag import RAGEngine
from src.model import load_model_for_inference
from src.utils import get_config, format_prompt, get_stop_token_ids


def main():
    """
    Main function to run the RAG evaluation.
    """

    # The RAGAS judge needs OpenAI: fail now, not after loading the model and generating
    load_dotenv(dotenv_path=find_dotenv(usecwd=True), override=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (env or .env); RAGAS needs it as the judge.")

    # --- 1. Load Configuration and Models ---
    print("Loading configuration and models...")
    config = get_config()
    output_dir = config['training']['output_dir']
    model_path = f"{output_dir}/final_model"

    model, tokenizer = load_model_for_inference(config, model_path)
    print("Models loaded successfully.")

    # --- 2. Initialize RAG Engine and Build Index ---
    print("Initializing RAG Engine...")
    rag_engine = RAGEngine()

    all_docs = []
    for doc_path in config['rag']['documents']:
        all_docs += rag_engine.load_documents(doc_path)

    rag_engine.build_index(all_docs)
    print("RAG Engine initialized and index built.")

    # --- 3. Define Golden Dataset ---
    # PakWheels questions whose answers are in data/pakwheels/knowledge_base.md
    golden_dataset = {
        "question": [
            "How much does it cost to feature my car ad for 14 days?",
            "What are the Sell It For Me charges for a 1300cc car?",
            "Which cities is PakWheels car inspection available in?",
            "Can I return car accessories if I change my mind?",
            "What are PakWheels support hours?",
        ],
        "ground_truth": [
            "Featuring one ad for 14 days costs PKR 4,450.",
            "A 1300cc car has a non-refundable onboarding fee of PKR 5,000, plus a 1% commission "
            "on the selling price after the sale (minimum PKR 5,000 if sold for 5 lakh or less).",
            "Karachi, Lahore, Islamabad, Rawalpindi, Peshawar, Faisalabad, Gujranwala, Gujrat, "
            "Hyderabad, Multan, Sargodha and Sialkot.",
            "No. Change of mind is not accepted for car or bike accessories; returns are accepted "
            "for damaged, defective, incomplete, wrong or mismatched items within 3 business days.",
            "Human support is available Monday to Sunday, 9 am to 9 pm, on 042-111-943-357.",
        ],
    }

    dataset = Dataset.from_dict(golden_dataset)

    # --- 4. Generate Answers ---
    print("Generating answers for the golden dataset...")
    results = []

    for entry in dataset:
        question = entry["question"]

        retrieved_context = build_context(question, [], rag_engine.retrieve(
            question, k=config['rag']['top_k'], min_similarity=config['rag']['min_similarity']
        ))
        max_new_tokens = 128
        prompt = format_prompt(
            tokenizer, question, history=[], context=retrieved_context,
            max_prompt_tokens=config['model']['max_seq_length'] - max_new_tokens,
        )

        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=get_stop_token_ids(tokenizer),
        )

        # Decode only the newly generated tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = guard(tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
                       retrieved_context, question)

        results.append(
            {
                "question": question,
                "answer": answer,
                "contexts": [retrieved_context],
                "ground_truth": entry["ground_truth"],
            }
        )

    results_dataset = Dataset.from_list(results)
    print("Answer generation complete.")

    # --- 5. Evaluate with RAGAS ---
    print("Evaluating generated answers with RAGAS...")

    metrics = [
        context_precision,
        faithfulness,
        answer_relevancy,
    ]

    # Explicit OpenAI client (used ONLY for evaluation)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=client,
    )

    # raise_exceptions=True: by default RAGAS turns a failed judge call into a
    # NaN score, so a broken run would still look complete
    result = evaluate(
        dataset=results_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        raise_exceptions=True,
    )

    print("Evaluation complete.")
    print("--- RAGAS Evaluation Results ---")
    print(result)


if __name__ == "__main__":
    main()
