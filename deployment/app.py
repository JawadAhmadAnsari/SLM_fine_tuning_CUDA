# deployment/app.py

import gradio as gr
from threading import Thread
import os
import sys
from transformers import TextIteratorStreamer


# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (format_prompt, get_config, get_stop_token_ids, recent_history,
                       retrieval_query)
from src.data import PLACEHOLDER_RE
from src.facts import FIGURE_RE, build_context, guard
from src.model import load_model_for_inference
from src.rag import RAGEngine

# --- Configuration & Model Loading ---
config = get_config()
output_dir = config['training']['output_dir']
model_path = f"{output_dir}/final_model"
model, tokenizer = load_model_for_inference(config, model_path)
MAX_SEQ_LENGTH = config['model']['max_seq_length']
MAX_NEW_TOKENS = 256
MAX_HISTORY_TURNS = config['deployment'].get('max_history_turns', 2)

# --- RAG Engine Initialization ---
# Index the knowledge base listed in config (rag.documents): the PakWheels KB by default
rag_engine = RAGEngine()
all_docs = []
for doc_path in config['rag']['documents']:
    all_docs += rag_engine.load_documents(doc_path)
if not all_docs:
    raise FileNotFoundError(f"No RAG documents loaded from {config['rag']['documents']}")
rag_engine.build_index(all_docs)


# --- Inference Function ---
def run_inference(message, history):
    """
    Generates a response from the model based on the user's message and conversation history.
    """
    # Only the last few exchanges: older answers otherwise leak into new ones
    history = recent_history(history, MAX_HISTORY_TURNS)
    # Empty when nothing is relevant enough; the prompt then has no Context block.
    # Follow-ups ("and for a Prado?") are searched together with the previous question.
    retrieved = rag_engine.retrieve(
        retrieval_query(message, history),
        k=config['rag']['top_k'], min_similarity=config['rag']['min_similarity'],
    )
    # Prices and city/eligibility verdicts come from src/facts.py, not the model's memory
    context = build_context(message, history, retrieved)
    # Trim old turns / context so prompt + reply fit in max_seq_length
    prompt = format_prompt(
        tokenizer, message, history, context,
        max_prompt_tokens=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    # Use a TextIteratorStreamer for real-time, token-by-token output.
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Run generation in a separate thread
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,  # same greedy decoding as scripts/demo_eval.py
        eos_token_id=get_stop_token_ids(tokenizer),
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # ChatInterface expects the full response so far on each yield.
    # Safety net: never show a leftover {{Placeholder}} to a customer.
    # Once the reply quotes a price or percentage, the rest is held back until the
    # guard has checked every figure against the context, so a wrong one never shows.
    response = ""
    holding = False
    for new_text in streamer:
        response += new_text
        holding = holding or bool(FIGURE_RE.search(response))
        if not holding:
            yield PLACEHOLDER_RE.sub(r"\1", response)
    yield PLACEHOLDER_RE.sub(r"\1", guard(response, context, message, history))

# --- Gradio UI ---
HELPLINE = "042-111-943-357"

HEADER_HTML = f"""
<div class="pw-header">
  <h1>PakWheels Car Support</h1>
  <p class="pw-lede">Questions about buying or selling a car, your ads and account, inspections,
  Sell It For Me, auction sheets, finance, insurance or AutoStore orders. Car and PakWheels
  topics only.</p>
  <div class="pw-status">
    <span><b>AI assistant</b> answers 24/7</span>
    <span><b>Human agents</b> 9 am to 9 pm daily, {HELPLINE}</span>
  </div>
</div>
"""

FOOTER_HTML = """
<p class="pw-disclaimer">Unofficial demo, not affiliated with or endorsed by PakWheels.
Answers can be wrong; confirm prices, fees and policies on pakwheels.com.
Never share your password, OTP or card details in this chat.</p>
"""

EXAMPLES = [
    "How do I post an ad for my car?",
    "How much does a featured ad cost?",
    "What does PakWheels inspection check?",
    "Sell It For Me charges kya hain?",
    "How do I verify an auction sheet?",
    "The seller is asking for an advance payment. What should I do?",
]

CSS = """
.gradio-container { font-family: 'Saira', 'Segoe UI', system-ui, sans-serif !important;
                    max-width: 860px !important; margin: 0 auto !important; }
.pw-header { background: #16181d; color: #f4f1ea; border-radius: 14px;
             padding: 22px 26px 18px; border-left: 6px solid #f2a516; }
.pw-header h1 { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.01em;
                margin: 0 0 6px; color: #f4f1ea; }
.pw-lede { margin: 0 0 14px; max-width: 64ch; line-height: 1.5; color: #c9c5bb; }
.pw-status { display: flex; flex-wrap: wrap; gap: 8px 20px; font-size: 0.92rem; color: #c9c5bb; }
.pw-status b { color: #f2a516; font-weight: 600; }
.pw-disclaimer { font-size: 0.82rem; color: #6b6f78; text-align: center;
                 max-width: 70ch; margin: 10px auto 0; line-height: 1.5; }
"""

HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Saira:wght@400;600;700&display=swap" '
    'rel="stylesheet">'
)

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_md,
)

with gr.Blocks(title="PakWheels Car Support (demo)") as demo:
    gr.HTML(HEADER_HTML)
    gr.ChatInterface(
        fn=run_inference,
        chatbot=gr.Chatbot(
            height=460,
            placeholder="Ask anything about cars on PakWheels: selling, buying, inspection, "
                        "auction sheets, finance or your AutoStore order.",
        ),
        textbox=gr.Textbox(placeholder="Type your car or PakWheels question...",
                           submit_btn=True),
        examples=EXAMPLES,
    )
    gr.HTML(FOOTER_HTML)

if __name__ == "__main__":
    # Gradio 6: theme, css and head are launch() arguments
    launch_kwargs = dict(theme=THEME, css=CSS, head=HEAD)
    # Launch with sharing enabled in Google Colab
    if "google.colab" in sys.modules:
        demo.launch(share=True, **launch_kwargs)
    else:
        demo.launch(**launch_kwargs)
