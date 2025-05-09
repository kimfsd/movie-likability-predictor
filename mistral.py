import pandas as pd
import ast
import requests
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Your API Key
MISTRAL_API_KEY = 'please'  # <<< Replace with your key

# Helper to safely parse lists
def safe_literal_eval(x):
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except:
        return []

# Load the cleaned data
df = pd.read_csv('final_movie_reviews.csv')

# Safely parse the stringified lists
df['movies_reviewed'] = df['movies_reviewed'].apply(safe_literal_eval)
df['ratings'] = df['ratings'].apply(safe_literal_eval)
df['overviews'] = df['overviews'].apply(safe_literal_eval)

print(f"Loaded {len(df)} users.")

# Only keep the first 50 users
selected_users = df['user_name'].unique()[:50]
df = df[df['user_name'].isin(selected_users)].reset_index(drop=True)

# Flatten the dataset
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
        if isinstance(o, str) and len(o.strip()) > 0:  # only keep real overviews
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

print(full_reviews_df.head())
print(f"Flattened to {len(full_reviews_df)} reviews total.")

# Optional: sample if too many (for faster testing)
if len(full_reviews_df) > 500:
    full_reviews_df = full_reviews_df.sample(n=500, random_state=42).reset_index(drop=True)
else:
    full_reviews_df = full_reviews_df.reset_index(drop=True)

# Train/test split
train_df, test_df = train_test_split(full_reviews_df, test_size=0.2, random_state=42, stratify=full_reviews_df['user'])

# Build user history dictionary
user_history = {}
for idx, row in train_df.iterrows():
    u = row['user']
    movie = row['movie']
    rating = row['rating']
    if u not in user_history:
        user_history[u] = []
    user_history[u].append((movie, rating))

# Prompt builder
def build_prompt(user_movies, user_ratings, new_movie_overview):
    history_snippets = []
    for movie, rating in zip(user_movies, user_ratings):
        sentiment = "liked" if rating == 1 else "disliked"
        history_snippets.append(f'"{movie}" ({sentiment})')
    history_text = ", ".join(history_snippets)
    prompt = (
        f"The user has previously {history_text}.\n\n"
        f"Here is a new movie description:\n\n"
        f"\"{new_movie_overview}\"\n\n"
        f"Will the user like this movie? Answer only with 0 (dislike) or 1 (like)."
    )
    return prompt

# Call Mistral API
def query_mistral(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-small",  # or mistral-medium if you have access
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1  # small because we only need 0 or 1
    }
    try:
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        output = response.json()
        answer = output['choices'][0]['message']['content'].strip()
        if answer not in ['0', '1']:
            return random.choice([0, 1])
        return int(answer)
    except Exception as e:
        print(f"Error calling Mistral API: {e}")
        return random.choice([0, 1])

# --- Evaluation
y_true = []
y_pred = []

for idx, row in test_df.iterrows():
    user = row['user']
    overview = row['overview']
    true_label = row['rating']
    
    if user not in user_history or len(user_history[user]) < 3:
        continue

    # Sample up to 5 past movies
    past_movies = random.sample(user_history[user], min(5, len(user_history[user])))
    movies_only = [x[0] for x in past_movies]
    ratings_only = [x[1] for x in past_movies]

    prompt = build_prompt(movies_only, ratings_only, overview)
    pred_label = query_mistral(prompt)

    y_true.append(true_label)
    y_pred.append(pred_label)

# --- Results
accuracy = accuracy_score(y_true, y_pred)
print(f"\nAccuracy on Test Set: {accuracy:.4f}")
