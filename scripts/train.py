# scripts/train.py
import builtins
import psutil

builtins.psutil = psutil  
import sys
import os

# Add the src directory to the Python path
# This allows us to import modules from src like `from train import train`
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train import train

if __name__ == "__main__":
    print("Executing training script...")
    train()
    print("Training script finished.")