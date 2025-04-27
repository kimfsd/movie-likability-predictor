# maybe we can try something with mistral

import pandas as pd
import random

from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

csv_path = 'final_movie_reviews.csv'
df = pd.read_csv(csv_path)

df['movies_reviewed'] = df['movies_reviewed'].apply(eval)
df['ratings'] = df['ratings'].apply(eval)
df['overviews'] = df['overviews'].apply(eval)

print(f"Loaded {len(df)} users.")

user_list = []
movie_list = []
rating_list = []
overview_list = []

for idx, row in df.iterrows():
    user = row['users']
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

#debugging
full_reviews_df = full_reviews_df.sample(n=500, random_state=42).reset_index(drop=True)

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
    """
    Create a prompt to send to Mistral.
    """

    history_snippets = []
    
    for movie, rating in history[-5:]:
        sentiment = "liked" if rating == 1 else "disliked"
        history_snippets.append(f'"{movie}" ({sentiment})')
    
    history_text = ", ".join(history_snippets)
    
    prompt = (
        f"The user has previously {history_text}.\n"
        f"Here is a new movie description:\n\n"
        f"\"{new_movie_overview}\"\n\n"
        f"Will the user like this movie? Answer only with 0 (dislike) or 1 (like)."
    )
    return prompt

def fake_mistral_response(prompt):
    """
    This function fakes a Mistral prediction randomly.
    Replace this with your actual API call to Mistral!
    """
    return random.choice([0, 1])

y_true = []
y_pred = []

for idx, row in test_df.iterrows():
    user = row['user']
    overview = row['overview']
    true_label = row['rating']
    
    if user not in user_history:
        continue
    
    prompt = build_prompt(user, user_history[user], overview)
    pred_label = fake_mistral_response(prompt)
    
    y_true.append(true_label)
    y_pred.append(pred_label)

#evaluate
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy on Test Set: {accuracy:.4f}")

