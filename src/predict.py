import torch
from transformers import AutoTokenizer
from src.models.model import GrievanceClassifier
import numpy as np
from src.data.preprocessing import clean_text

class GrievancePredictor:
    def __init__(self, model_path='best_model_state.bin', category_map_path='category_map.npy', priority_map_path='priority_map.npy'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.category_map = np.load(category_map_path, allow_pickle=True).item()
        self.priority_map = np.load(priority_map_path, allow_pickle=True).item()
        self.inv_category_map = {v: k for k, v in self.category_map.items()}
        self.inv_priority_map = {v: k for k, v in self.priority_map.items()}
        
        self.tokenizer = AutoTokenizer.from_pretrained('google/muril-base-cased')
        self.MAX_LEN = 128
        
        self.model = GrievanceClassifier(len(self.category_map), len(self.priority_map))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, text):
        clean_txt = clean_text(text)
        encoded_text = self.tokenizer.encode_plus(
            clean_txt,
            max_length=self.MAX_LEN,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)
        
        with torch.no_grad():
            category_logits, priority_logits = self.model(input_ids, attention_mask)
            
            probs = torch.nn.functional.softmax(category_logits, dim=1)
            confidence_score, category_pred = torch.max(probs, dim=1)
            
            _, priority_pred = torch.max(priority_logits, dim=1)
            
            category = self.inv_category_map[category_pred.item()]
            priority = self.inv_priority_map[priority_pred.item()]
            confidence = confidence_score.item()

            # --- RULE BASED OVERRIDES ---
            text_lower = text.lower()
            
            # 1. Revenue Department Keywords
            REVENUE_KEYWORDS = ['lekhpal', 'tehsildar', 'patwari', 'kanungo', 'naib tehsildar', 'sdm', 'chakbandi']
            for kw in REVENUE_KEYWORDS:
                if kw in text_lower:
                    category = 'Revenue & Disaster Management'
                    confidence = 1.0 # High confidence for rule-based match
                    break
            
            # 2. Rename Category
            if category == 'Sanitation & Water (Jal Nigam/Nagar Nigam)':
                category = 'Sanitation & Water (Jal Nigam)'
            
            return category, priority, confidence
if __name__ == "__main__":
    # Example usage
    predictor = GrievancePredictor()
    text = "There is no water supply in my area since yesterday."
    cat, prio, conf = predictor.predict(text)
    print(f"Grievance: {text}")
    print(f"Category: {cat}")
    print(f"Priority: {prio}")
