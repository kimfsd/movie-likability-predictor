import pandas as pd
import random
import requests
import ast
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = "bQoGVfrxvPCtTqLksBuKmbfucepEsvjq"

csv_path = 'final_movie_reviews.csv'
df = pd.read_csv(csv_path)

def safe_eval(val):
    if pd.isna(val):
        return []
    val = str(val).replace('nan', 'None')
    return ast.literal_eval(val)

df['movies_reviewed'] = df['movies_reviewed'].apply(safe_eval)
df['ratings'] = df['ratings'].apply(safe_eval)
df['overviews'] = df['overviews'].apply(safe_eval)

print(f"Loaded {len(df)} users.")

user_list, movie_list, rating_list, overview_list = [], [], [], []

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

print(f"Flattened to {len(full_reviews_df)} reviews total.")

train_df, test_df = train_test_split(full_reviews_df, test_size=0.2, random_state=42, stratify=full_reviews_df['user'])
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

user_history = {}

for idx, row in train_df.iterrows():
    u = row['user']
    movie = row['movie']
    rating = row['rating']
    if u not in user_history:
        user_history[u] = []
    user_history[u].append((movie, rating))

def build_prompt(user, history, new_movie_overview):
    history_snippets = []
    for movie, rating in history[-5:]:  # Last 5 movies
        sentiment = "liked" if rating == 1 else "disliked"
        history_snippets.append(f'"{movie}" ({sentiment})')
    
    history_text = ", ".join(history_snippets)
    
    prompt = (
        f"The user has previously {history_text}.\n"
        f"Here is a new movie description:\n\n"
        f"\"{new_movie_overview}\"\n\n"
        f"Will the user like this movie? Think carefully, then answer only with 0 (dislike) or 1 (like)."
    )
    return prompt

def call_mistral(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,  # No randomness
    }
    
    response = requests.post(MISTRAL_API_URL, json=body, headers=headers)
    response.raise_for_status()
    output = response.json()
    answer = output['choices'][0]['message']['content'].strip()

    if '1' in answer and '0' not in answer:
        return 1
    elif '0' in answer and '1' not in answer:
        return 0
    else:
        return random.choice([0, 1])

y_true = []
y_pred = []

for idx, row in test_df.iterrows():
    user = row['user']
    overview = row['overview']
    true_label = row['rating']
    
    if user not in user_history:
        continue  # Skip users with no history
    
    prompt = build_prompt(user, user_history[user], overview)
    
    try:
        pred_label = call_mistral(prompt)
    except Exception as e:
        print(f"Error calling Mistral API: {e}")
        continue
    
    y_true.append(true_label)
    y_pred.append(pred_label)

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy on Test Set: {accuracy:.4f}")
