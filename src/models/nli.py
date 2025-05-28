"""
NLI model management for company classification.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class NLIManager:
    """Manages NLI model and classification."""
    
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load NLI model and tokenizer."""
        print(f"  - Loading NLI Model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print(f"    NLI model '{self.model_name}' loaded successfully.")
            return True
        except Exception as e:
            print(f"    Error loading NLI model '{self.model_name}': {e}")
            return False
    
    def classify(self, premise_text, label_name_text):
        """Classify using NLI."""
        hypothesis = f"The company's operations and main activities are related to '{label_name_text}'."
        try:
            premise_text_str = str(premise_text) if premise_text is not None else ""
            inputs = self.tokenizer(
                premise_text_str, hypothesis, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
            
            ent_id = self.model.config.label2id.get('entailment', -1)
            neut_id = self.model.config.label2id.get('neutral', -1)
            contr_id = self.model.config.label2id.get('contradiction', -1)
            
            res = {'entailment': 0.0, 'neutral': 0.0, 'contradiction': 1.0}
            if ent_id != -1:
                res['entailment'] = probabilities[0][ent_id].item()
            if neut_id != -1:
                res['neutral'] = probabilities[0][neut_id].item()
            if contr_id != -1:
                res['contradiction'] = probabilities[0][contr_id].item()
            
            return res
        except Exception as e:
            print(f"Error during NLI classification: {e}")
            return {'entailment': 0.0, 'neutral': 0.0, 'contradiction': 1.0} 