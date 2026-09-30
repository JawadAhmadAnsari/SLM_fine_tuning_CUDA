# deployment/app.py

import gradio as gr
from threading import Thread
import os
import sys
from transformers import TextIteratorStreamer


# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import format_prompt, get_config, get_stop_token_ids
from src.data import PLACEHOLDER_RE
from src.model import load_model_for_inference
from src.rag import RAGEngine

# --- Configuration & Model Loading ---
config = get_config()
output_dir = config['training']['output_dir']
model_path = f"{output_dir}/final_model"
model, tokenizer = load_model_for_inference(config, model_path)
MAX_SEQ_LENGTH = config['model']['max_seq_length']
MAX_NEW_TOKENS = 256

# --- RAG Engine Initialization ---
rag_engine = RAGEngine()
# Load documents from both the product catalog and employee handbook
product_docs = rag_engine.load_documents("data/product_catalog.xlsx")
handbook_docs = rag_engine.load_documents("data/employee_handbook.pdf")
all_docs = product_docs + handbook_docs
rag_engine.build_index(all_docs)


# --- Inference Function ---
def run_inference(message, history):
    """
    Generates a response from the model based on the user's message and conversation history.
    """
    # Empty when nothing is relevant enough; the prompt then has no Context block
    context = rag_engine.retrieve(
        message, k=config['rag']['top_k'], min_similarity=config['rag']['min_similarity']
    )
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
        eos_token_id=get_stop_token_ids(tokenizer),
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # ChatInterface expects the full response so far on each yield.
    # Safety net: never show a leftover {{Placeholder}} to a customer.
    response = ""
    for new_text in streamer:
        response += new_text
        yield PLACEHOLDER_RE.sub(r"\1", response)

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown(f"# Customer Support SLM")
    
    gr.ChatInterface(
        fn=run_inference,
        title="Customer Support Chatbot",
        description="Ask a question, and the AI agent will respond based on its training.",
        examples=[
            ["How do I reset my password?"],
            ["What is your return policy?"],
            ["I can't log in to my account."],
        ]
    )

if __name__ == "__main__":
    # Launch with sharing enabled in Google Colab
    if "google.colab" in sys.modules:
        demo.launch(share=True)
    else:
        demo.launch()
