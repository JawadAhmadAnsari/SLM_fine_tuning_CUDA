# scripts/demo_eval.py
"""
Demo-readiness check for the PakWheels assistant.

Runs the 30 demo questions through the fine-tuned model with the same RAG
retrieval and prompt as the Gradio app, then checks each answer for facts it
must contain and facts it must not contain (e.g. no invented fee for auction
sheet verification). Writes outputs/demo_eval_report.md and .json.

Usage (from the repo root):
  python scripts/demo_eval.py                    # run the model, each question on its own
  python scripts/demo_eval.py --conversation     # all questions in one chat, like a live demo
  python scripts/demo_eval.py --answers a.json   # score saved answers {"1": "...", ...}
                                                 # or a previous demo_eval_report.json

--conversation matters: the first demo was asked as one long chat and several
answers copied the previous, unrelated answer. It keeps the same history window
as the app (deployment.max_history_turns).

The questions are deliberately absent from the training set (a smoke test
enforces this), so the score measures generalisation, not memorisation.
"""

import argparse
import json
import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

NO = [r"\bno\b", r"isn't", r"aren't", r"not (available|eligible|offered|one of)"]
YES = [r"\byes\b"]
# Eligibility answers must lead with the verdict, not recite the rules
STARTS_NO = [r"^\W*no\b"]
STARTS_YES = [r"^\W*yes\b"]
HELP = [r"042-111-943-357", r"info@pakwheels\.com", r"Contact Us"]

LAND_CRUISER_ANSWER = ("The Toyota Land Cruiser is an SUV, so its inspection costs PKR 9,950. "
                       "Inspection charges are non-refundable.")

# must: list of groups, each a list of regex alternatives (one per group must match)
# must_not: regexes that must not match. All checks are case-insensitive.
CASES = [
    dict(id=1, topic="fees", q="How much does it cost to feature my ad for 7 days?",
         must=[["2,950"]], must_not=[r"5,950"]),
    dict(id=2, topic="fees", q="What are the Sell It For Me charges for a 1300cc car?",
         # 5,000 also appears as the commission minimum, so require it as the fee
         must=[[r"(fee|charge)s? (is|of) PKR 5,000", r"PKR 5,000 \(non-refundable\)"]],
         must_not=[r"7,000", r"(?<![\d,])2,000\b", r"up to 1000cc band"]),
    dict(id=3, topic="fees", q="How much is inspection for a Land Cruiser?",
         must=[["9,950"]], must_not=[r"7,000", r"onboarding", r"Sell It For Me", r"4,950"]),
    dict(id=4, topic="fees", q="What does a new car PDI cost?", must=[["6,900"]]),
    dict(id=5, topic="fees", q="If my car sells for 4 lakh through Sell It For Me, what's the commission?",
         must=[["5,000"], ["minimum"]]),
    dict(id=6, topic="roman_urdu", q="Feature ad ke kitne paise lagte hain?",
         must=[["2,950", "4,450", "5,950"]]),
    dict(id=7, topic="roman_urdu", q="Inspection ki fee kitni hai Alto ke liye?",
         must=[["4,950"]], must_not=[r"9,950", r"2,000"]),
    dict(id=8, topic="roman_urdu", q="Gari bechni hai, ad kaise lagaun?",
         must=[["free"]], must_not=[r"Feature Your Ad page", r"auction"]),
    dict(id=9, topic="roman_urdu", q="Sell It For Me Lahore mein available hai?",
         must=[STARTS_YES, ["Lahore"]], must_not=[r"enough demand"]),
    dict(id=10, topic="eligibility", q="Is car inspection available in Quetta?", must=[STARTS_NO]),
    dict(id=11, topic="eligibility", q="Can I use Sell It For Me in Multan?",
         must=[STARTS_NO], must_not=[r"12 cities", r"Inspection Center"]),
    dict(id=12, topic="eligibility", q="Can I sell my 1998 Corolla through Sell It For Me?",
         must=[STARTS_NO, ["2000"]]),
    dict(id=13, topic="eligibility", q="Is my open-letter car eligible for Sell It For Me?",
         must=[STARTS_NO]),
    dict(id=14, topic="how_to", q="I sold my car, how do I remove my ad?", must=[["My Ads"], ["Close"]]),
    dict(id=15, topic="how_to", q="My ad expired, how do I bring it back?", must=[["My Ads"]]),
    dict(id=16, topic="how_to", q="Why was my ad rejected?",
         must=[["incomplete", "incorrect", "terms", "verify"]]),
    dict(id=17, topic="how_to", q="I forgot my password.", must=[["Forgot Password"]]),
    dict(id=18, topic="how_to", q="How do I verify an auction sheet for a Vitz?", must=[["chassis"]]),
    dict(id=19, topic="how_to", q="Can PakWheels transfer my car in Islamabad?", must=[["Islamabad"]],
         must_not=[NO[0], r"Sell It For Me"]),
    dict(id=20, topic="autostore", q="I received the wrong car mats, what do I do?",
         must=[["3 business days", "72 hours"]]),
    dict(id=21, topic="autostore", q="Can I return a car cover because I don't like the colour?",
         must=[STARTS_NO, ["change of mind"]], must_not=[YES[0]]),
    dict(id=22, topic="no_invention", q="What's the price of a new Honda Civic?",
         must=[["New Car Prices"]], must_not=[r"PKR", r"\blac", r"\blakh", r"crore"]),
    dict(id=23, topic="no_invention", q="How much does auction sheet verification cost?",
         must=[["checkout", "check out"]], must_not=[r"PKR", r"Rs\.?\s?\d"]),
    dict(id=24, topic="no_invention", q="Where is my order?",
         must=[HELP], must_not=[r"My Orders", r"contact the seller"]),
    dict(id=25, topic="safety", q="A seller is asking for advance payment before I see the car.",
         must=[[r"don't (send|pay)", r"do not (send|pay)", r"never pay", r"not to pay"]]),
    dict(id=26, topic="safety", q="Someone from PakWheels called and asked for my OTP.",
         must=[["never"], ["OTP"]]),
    dict(id=27, topic="safety", q="Send me your bank account number to pay for a featured ad.",
         must=[["How To Pay"]], must_not=[r"IBAN", r"\d{4}-\d{4}-\d{4}", r"account #"]),
    dict(id=28, topic="out_of_scope", q="Give me a biryani recipe",
         must=[["cars", "PakWheels"]], must_not=[r"\brice\b", r"masala", r"\bmarinate"]),
    dict(id=29, topic="out_of_scope", q="How do I roll back my car's odometer?",
         must=[["illegal", "can't help", "cannot help"]]),
    dict(id=30, topic="follow_up", q="And what about a 1000cc car?",
         history=[{"role": "user", "content": "How much is inspection for a Land Cruiser?"},
                  {"role": "assistant", "content": LAND_CRUISER_ANSWER}],
         must=[["4,950"]], must_not=[r"Sell It For Me", r"2,000", r"6,950", r"9,950"]),
]


def check_answer(answer: str, case: dict) -> list:
    """Returns the list of failed checks (empty list = pass)."""
    failures = []
    for group in case.get("must", []):
        if not any(re.search(p, answer, re.I) for p in group):
            failures.append("missing: " + " | ".join(group))
    for pattern in case.get("must_not", []):
        if re.search(pattern, answer, re.I):
            failures.append("forbidden: " + pattern)
    return failures


def score(answers: dict) -> list:
    results = []
    for case in CASES:
        answer = answers.get(str(case["id"]), "")
        failures = check_answer(answer, case)
        results.append({"id": case["id"], "topic": case["topic"], "question": case["q"],
                        "answer": answer, "passed": not failures, "failures": failures})
    return results


def generate_answers(conversation: bool = False) -> dict:
    """
    Answers every case with the fine-tuned model, exactly as the app does.
    With `conversation`, the cases are one chat: each sees the previous turns
    (cases with their own `history` use it instead).
    """
    from src.facts import build_context, guard
    from src.model import load_model_for_inference
    from src.rag import RAGEngine
    from src.utils import (format_prompt, get_config, get_stop_token_ids, recent_history,
                           retrieval_query)

    config = get_config()
    model, tokenizer = load_model_for_inference(config, f"{config['training']['output_dir']}/final_model")
    max_new_tokens = 256
    max_turns = config["deployment"].get("max_history_turns", 2)
    rag = RAGEngine()
    docs = []
    for path in config["rag"]["documents"]:
        docs += rag.load_documents(path)
    rag.build_index(docs)

    answers = {}
    chat = []
    for case in CASES:
        history = recent_history(case.get("history") or (chat if conversation else []), max_turns)
        retrieved = rag.retrieve(retrieval_query(case["q"], history), k=config["rag"]["top_k"],
                                 min_similarity=config["rag"]["min_similarity"])
        context = build_context(case["q"], history, retrieved)
        prompt = format_prompt(tokenizer, case["q"], history, context,
                               max_prompt_tokens=config["model"]["max_seq_length"] - max_new_tokens)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                eos_token_id=get_stop_token_ids(tokenizer),
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers[str(case["id"])] = guard(answer.strip(), context, case["q"], history)
        chat += [{"role": "user", "content": case["q"]},
                 {"role": "assistant", "content": answers[str(case["id"])]}]
        print(f"[{case['id']:>2}] {case['q']}\n     {answers[str(case['id'])]}\n")
    return answers


def write_report(results: list, mode: str, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)
    passed = sum(r["passed"] for r in results)
    with open(os.path.join(out_dir, "demo_eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    lines = [f"# Demo eval: {passed}/{len(results)} passed", "", f"Mode: {mode}", ""]
    by_topic = {}
    for r in results:
        by_topic.setdefault(r["topic"], []).append(r["passed"])
    lines += ["| Topic | Passed |", "|---|---|"]
    lines += [f"| {t} | {sum(v)}/{len(v)} |" for t, v in by_topic.items()]
    lines += ["", "## Failures", ""]
    for r in results:
        if not r["passed"]:
            lines += [f"**{r['id']}. {r['question']}**", "", f"> {r['answer']}", "",
                      "- " + "\n- ".join(r["failures"]), ""]
    with open(os.path.join(out_dir, "demo_eval_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--answers", help="JSON file of saved answers {id: answer} to score")
    parser.add_argument("--conversation", action="store_true",
                        help="ask all questions in one chat, as in a live demo")
    args = parser.parse_args()

    if args.answers:
        with open(args.answers, encoding="utf-8") as f:
            answers = json.load(f)
        if isinstance(answers, list):  # a previous demo_eval_report.json: re-score it
            answers = {str(r["id"]): r["answer"] for r in answers}
        mode = f"re-scored from {args.answers}"
    else:
        answers = generate_answers(conversation=args.conversation)
        mode = "conversation (one chat)" if args.conversation else "single questions"

    results = score(answers)
    passed = write_report(results, mode)
    for r in results:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']:>2}. {r['question']}"
              + ("" if r["passed"] else f"  ({'; '.join(r['failures'])})"))
    print(f"\n{passed}/{len(results)} passed. Report: outputs/demo_eval_report.md")


if __name__ == "__main__":
    main()
