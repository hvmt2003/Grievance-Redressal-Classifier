import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from src.data.preprocessing import GrievanceDataset, clean_text
from src.models.model import GrievanceClassifier

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_category_predictions = 0
    correct_priority_predictions = 0
    
    for d in data_loader:
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
        
        loss_category = loss_fn(category_logits, categories)
        loss_priority = loss_fn(priority_logits, priorities)
        loss = loss_category + loss_priority
        
        correct_category_predictions += torch.sum(category_preds == categories)
        correct_priority_predictions += torch.sum(priority_preds == priorities)
        
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_category_predictions.double() / n_examples, correct_priority_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()
    losses = []
    correct_category_predictions = 0
    correct_priority_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
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
            
            loss_category = loss_fn(category_logits, categories)
            loss_priority = loss_fn(priority_logits, priorities)
            loss = loss_category + loss_priority
            
            correct_category_predictions += torch.sum(category_preds == categories)
            correct_priority_predictions += torch.sum(priority_preds == priorities)
            
            losses.append(loss.item())
            
    return correct_category_predictions.double() / n_examples, correct_priority_predictions.double() / n_examples, np.mean(losses)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()

    # Configuration
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = args.epochs
    DATA_PATH = os.path.join("data", "raw", "grievances.csv")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run make_dataset.py first.")
        
    df = pd.read_csv(DATA_PATH)
    df['text'] = df['text'].apply(clean_text)
    
    # Label Encoding
    category_map = {c: i for i, c in enumerate(df['category'].unique())}
    priority_map = {p: i for i, p in enumerate(df['priority'].unique())}
    
    df['category_code'] = df['category'].map(category_map)
    df['priority_code'] = df['priority'].map(priority_map)
    
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained('google/muril-base-cased')
    
    train_data_loader = DataLoader(
        GrievanceDataset(
            df_train.text.to_numpy(),
            df_train.category_code.to_numpy(),
            df_train.priority_code.to_numpy(),
            tokenizer,
            MAX_LEN
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
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
    model = GrievanceClassifier(len(category_map), len(priority_map))
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * EPOCHS
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        
        train_cat_acc, train_prio_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        
        print(f'Train loss {train_loss} Category accuracy {train_cat_acc} Priority accuracy {train_prio_acc}')
        
        val_cat_acc, val_prio_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )
        
        print(f'Val   loss {val_loss} Category accuracy {val_cat_acc} Priority accuracy {val_prio_acc}')
        
        if val_cat_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_cat_acc
            
            # Save mappings
            np.save('category_map.npy', category_map)
            np.save('priority_map.npy', priority_map)

if __name__ == "__main__":
    main()
