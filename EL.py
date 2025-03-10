import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# Load Pretrained BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Example training data
training_data = [
    ("UPS", "United Parcel Service", 1),
    ("UPS", "University of Maryland", 0),
    ("University of MD", "University of Maryland", 1),
    ("University of MD", "University of Louisville", 0),
]

# Convert data into DataFrame
df = pd.DataFrame(training_data, columns=["mention", "candidate", "label"])
class EntityLinkingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mention, candidate, label = self.data[idx]
        encoding = tokenizer(mention, candidate, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }

# Create Dataset & DataLoader
dataset = EntityLinkingDataset(training_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

from transformers import BertForSequenceClassification, AdamW
import torch.nn.functional as F

# Load BERT Model for Classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

# Set Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_fn = torch.nn.BCEWithLogitsLoss()  # Binary classification loss

def train_model(model, dataloader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze()
            
            # Compute loss
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

# Train Model
train_model(model, dataloader, optimizer, epochs=3)

def predict_entity(mention, candidates):
    """
    Predict the best entity for a given mention using the fine-tuned model.
    """
    model.eval()
    scores = []
    
    for candidate in candidates:
        encoding = tokenizer(mention, candidate, return_tensors="pt", padding=True, truncation=True, max_length=64)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            score = model(input_ids, attention_mask=attention_mask).logits.item()
        
        scores.append((candidate, score))
    
    # Sort candidates by score (higher is better)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# Example: Predict Best Entity
mention = "UPS"
candidates = ["United Parcel Service", "University of Maryland"]

predictions = predict_entity(mention, candidates)
print(predictions)

import random

# List of all entities in KB
kb_entities = ["United Parcel Service", "University of Maryland", "University of Louisville"]

def generate_negative_samples(positive_pairs):
    """
    Automatically generate negative samples by selecting incorrect candidates.
    """
    augmented_data = []
    
    for mention, correct_entity in positive_pairs:
        # Add positive example
        augmented_data.append((mention, correct_entity, 1))
        
        # Generate one negative sample (choose a random incorrect entity)
        wrong_entity = random.choice([e for e in kb_entities if e != correct_entity])
        augmented_data.append((mention, wrong_entity, 0))
    
    return augmented_data

# Example: Generate training data with negatives
training_data = generate_negative_samples([
    ("UPS", "United Parcel Service"),
    ("University of MD", "University of Maryland"),
])

print(training_data)

