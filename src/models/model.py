from transformers import AutoModel
import torch.nn as nn

class GrievanceClassifier(nn.Module):
    def __init__(self, n_categories, n_priorities):
        super(GrievanceClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('google/muril-base-cased')
        self.drop = nn.Dropout(p=0.3)
        self.category_out = nn.Linear(self.bert.config.hidden_size, n_categories)
        self.priority_out = nn.Linear(self.bert.config.hidden_size, n_priorities)
        
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[1] # pooled_output
        
        output = self.drop(pooled_output)
        
        category_logits = self.category_out(output)
        priority_logits = self.priority_out(output)
        
        return category_logits, priority_logits
