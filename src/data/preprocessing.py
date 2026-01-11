import torch
from torch.utils.data import Dataset
import re

class GrievanceDataset(Dataset):
    def __init__(self, texts, categories, priorities, tokenizer, max_len):
        self.texts = texts
        self.categories = categories
        self.priorities = priorities
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        category = self.categories[item]
        priority = self.priorities[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'category': torch.tensor(category, dtype=torch.long),
            'priority': torch.tensor(priority, dtype=torch.long)
        }

def clean_text(text):
    text = text.lower()
    # Keep alphanumeric characters (in any language) and whitespace
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    return text
