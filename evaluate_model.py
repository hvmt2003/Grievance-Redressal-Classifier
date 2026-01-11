
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from src.data.preprocessing import GrievanceDataset, clean_text
from src.models.model import GrievanceClassifier
from tqdm import tqdm

def eval_model(model, data_loader, device, n_examples):
    model.eval()
    correct_category_predictions = 0
    correct_priority_predictions = 0
    
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            categories = d["category"].to(device)
            priorities = d["priority"].to(device)
            
            category_logits, priority_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, category_preds = torch.max(category_logits, dim=1)
            _, priority_preds = torch.max(priority_logits, dim=1)
            
            correct_category_predictions += torch.sum(category_preds == categories)
            correct_priority_predictions += torch.sum(priority_preds == priorities)
            
    return correct_category_predictions.double() / n_examples, correct_priority_predictions.double() / n_examples

def main():
    # Configuration matches train.py
    MAX_LEN = 128
    BATCH_SIZE = 16
    DATA_PATH = os.path.join("data", "raw", "grievances.csv")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
        
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['text'] = df['text'].apply(clean_text)
    
    # Load mappings
    if os.path.exists('category_map.npy'):
        category_map = np.load('category_map.npy', allow_pickle=True).item()
        priority_map = np.load('priority_map.npy', allow_pickle=True).item()
    else:
        # Fallback if map files don't exist, recreate same as train.py
        category_map = {c: i for i, c in enumerate(df['category'].unique())}
        priority_map = {p: i for i, p in enumerate(df['priority'].unique())}

    df['category_code'] = df['category'].map(category_map)
    df['priority_code'] = df['priority'].map(priority_map)
    
    # Same split as train.py
    _, df_val = train_test_split(df, test_size=0.1, random_state=42)
    
    print(f"Validation set size: {len(df_val)}")

    tokenizer = AutoTokenizer.from_pretrained('google/muril-base-cased')
    
    val_data_loader = DataLoader(
        GrievanceDataset(
            df_val.text.to_numpy(),
            df_val.category_code.to_numpy(),
            df_val.priority_code.to_numpy(),
            tokenizer,
            MAX_LEN
        ),
        batch_size=BATCH_SIZE
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = GrievanceClassifier(len(category_map), len(priority_map))
    
    if os.path.exists('best_model_state.bin'):
        print("Loading model weights...")
        model.load_state_dict(torch.load('best_model_state.bin', map_location=device))
    else:
        print("WARNING: best_model_state.bin not found! Using initialized weights (random).")

    model = model.to(device)
    
    cat_acc, prio_acc = eval_model(model, val_data_loader, device, len(df_val))
    
    print(f"\nValidation Accuracy:")
    print(f"Category Accuracy: {cat_acc.item():.4f} ({cat_acc.item()*100:.2f}%)")
    print(f"Priority Accuracy: {prio_acc.item():.4f} ({prio_acc.item()*100:.2f}%)")

if __name__ == "__main__":
    main()
