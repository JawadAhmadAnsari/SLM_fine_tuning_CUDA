# deployment/app.py

import gradio as gr
from threading import Thread
import os
import sys
from transformers import TextIteratorStreamer


# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import format_prompt, get_config
from src.model import load_model_for_inference
from src.rag import RAGEngine

# --- Configuration & Model Loading ---
config = get_config()
output_dir = config['training']['output_dir']
model_path = f"{output_dir}/final_model"
model, tokenizer = load_model_for_inference(config, model_path)

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
    context = rag_engine.retrieve(message)
    prompt = format_prompt(message, history, context)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    # Use a TextIteratorStreamer for real-time, token-by-token output.
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Run generation in a separate thread
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=256)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield partial responses from the streamer
    for new_text in streamer:
        yield new_text

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown(f"# Customer Support SLM")
    gr.Markdown(f"This interface is powered by the fine-tuned model from the `{model_path}` repository.")
    
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
