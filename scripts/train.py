# scripts/train.py
import sys
import os

# Add the project root to the Python path so `src.*` imports resolve
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train import train

if __name__ == "__main__":
    print("Executing training script...")
    train()
    print("Training script finished.")