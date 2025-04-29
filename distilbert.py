#distilBERT
#not using anymore

import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ast
import time

CSV_PATH = 'final_movie_reviews.csv'
MODEL_NAME = 'distilbert-base-uncased'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 512

print("Loading data...")

df = pd.read_csv(CSV_PATH)

def safe_eval(x):
    if pd.isna(x):
        return []
    x = x.strip()
    if not (x.startswith('[') and x.endswith(']')):
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []

df['movies_reviewed'] = df['movies_reviewed'].apply(safe_eval)
df['ratings'] = df['ratings'].apply(safe_eval)

# Flatten
user_list = []
movie_list = []
rating_list = []
overview_list = []

for idx, row in df.iterrows():
    user = row['user_name']
    movies = row['movies_reviewed']
    ratings = row['ratings']
    overviews = row['overviews']
    
    for m, r, o in zip(movies, ratings, overviews):
        user_list.append(user)
        movie_list.append(m)
        rating_list.append(r)
        overview_list.append(o)

full_reviews_df = pd.DataFrame({
    'user': user_list,
    'movie': movie_list,
    'rating': rating_list,
    'overview': overview_list
})

'''
LIMIT = 7000  # Set how many reviews you want to keep for quick experiments
if len(full_reviews_df) > LIMIT:
    full_reviews_df = full_reviews_df.sample(n=LIMIT, random_state=42).reset_index(drop=True)
print(f"After limiting, {len(full_reviews_df)} reviews used.")
'''

# Post-flatten filter: users must have >=10 reviews
user_counts = full_reviews_df['user'].value_counts()
valid_users = user_counts[user_counts >= 10].index
full_reviews_df = full_reviews_df[full_reviews_df['user'].isin(valid_users)].reset_index(drop=True)

user_history = {}
for idx, row in full_reviews_df.iterrows():
    u = row['user']
    m = row['movie']
    r = row['rating']
    if u not in user_history:
        user_history[u] = []
    user_history[u].append((m, r))

def build_prompt(user, new_movie_overview):
    history = user_history.get(user, [])
    prompt_parts = []
    for movie, rating in history[-5:]:
        sentiment = "liked" if rating == 1 else "disliked"
        prompt_parts.append(f'"{movie}" was {sentiment}')
    
    history_text = "; ".join(prompt_parts)
    
    prompt_text = (
        f"User's previous movie preferences: {history_text}.\n\n"
        f"Movie description: {new_movie_overview}\n\n"
        f"Based on this, predict if the user will like the movie. Answer 0 (dislike) or 1 (like)."
    )
    return prompt_text

full_reviews_df['prompt'] = full_reviews_df.apply(lambda row: build_prompt(row['user'], row['overview']), axis=1)

#train/test split
train_df, val_df = train_test_split(full_reviews_df, test_size=0.2, random_state=42, stratify=full_reviews_df['user'])

class MovieDataset(Dataset):
    def __init__(self, prompts, labels, tokenizer):
        self.prompts = prompts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(prompt, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_LENGTH)
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label)
        return item

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = MovieDataset(train_df['prompt'].tolist(), train_df['rating'].tolist(), tokenizer)
val_dataset = MovieDataset(val_df['prompt'].tolist(), val_df['rating'].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

#loading distilBERT model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.4f}")

    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = inputs.pop('labels')

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch+1}/{EPOCHS}, Validation Accuracy: {val_acc:.4f}")

timestamp = int(time.time())
model.save_pretrained(f'finetuned_distilbert_{timestamp}')
tokenizer.save_pretrained(f'finetuned_distilbert_{timestamp}')

print("Model saved successfully!")


