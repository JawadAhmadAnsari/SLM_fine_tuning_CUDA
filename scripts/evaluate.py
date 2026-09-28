# scripts/evaluate.py

import os
import sys
from datasets import Dataset
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

from src.rag import RAGEngine
from src.model import load_model_for_inference
from src.utils import get_config, format_prompt, get_stop_token_ids


def main():
    """
    Main function to run the RAG evaluation.
    """

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

    product_docs = rag_engine.load_documents("data/product_catalog.xlsx")
    handbook_docs = rag_engine.load_documents("data/employee_handbook.pdf")
    all_docs = product_docs + handbook_docs

    rag_engine.build_index(all_docs)
    print("RAG Engine initialized and index built.")

    # --- 3. Define Golden Dataset ---
    golden_dataset = {
        "question": [
            "What is the return policy for the Laptop Pro X?",
            "How much does the Smartwatch Series 5 cost?",
            "What are the features of the Drone Explorer?",
            "Can I use public Wi-Fi when working remotely?",
            "How do I get reimbursed for office supplies?",
        ],
        "ground_truth": [
            "The return policy for the Laptop Pro X is a 30-day money-back guarantee.",
            "The Smartwatch Series 5 costs $399.99.",
            "The Drone Explorer features a 4K camera and a 30-minute flight time.",
            "No, use of public Wi-Fi for sensitive work is strictly prohibited according to the remote work policy.",
            "To get reimbursed for office supplies, you must submit an expense report with original receipts within 30 days, after getting pre-approval from your manager.",
        ],
        "ground_truth_context": [
            "Product_ID: ELE-001, Name: Laptop Pro X, Price: 1299.99, Features: 16GB RAM, 512GB SSD, Intel i7, Return_Policy: 30-day money-back guarantee",
            "Product_ID: ELE-007, Name: Smartwatch Series 5, Price: 399.99, Features: GPS, Heart Rate Monitor, Return_Policy: 30-day money-back guarantee",
            "Product_ID: ELE-011, Name: Drone Explorer, Price: 899.99, Features: 4K camera, 30-min flight time, Return_Policy: 30-day money-back guarantee",
            "The company will provide necessary equipment, including a laptop and monitor. Employees are responsible for maintaining a secure and ergonomic home office setup. All company data must be handled in accordance with our data security policies. Use of public Wi-Fi for sensitive work is strictly prohibited.",
            "All work-related expenses must be pre-approved by your manager. To request reimbursement, submit an expense report with original receipts within 30 days of the purchase. Reimbursable expenses include travel, office supplies, and pre-approved software.",
        ],
    }

    dataset = Dataset.from_dict(golden_dataset)

    # --- 4. Generate Answers ---
    print("Generating answers for the golden dataset...")
    results = []

    for entry in dataset:
        question = entry["question"]

        retrieved_context = rag_engine.retrieve(question)
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
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

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

    try:
        result = evaluate(
            dataset=results_dataset,
            metrics=metrics,
            llm=evaluator_llm,
        )
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        result = {}

    print("Evaluation complete.")
    print("--- RAGAS Evaluation Results ---")
    print(result)


if __name__ == "__main__":
    main()
