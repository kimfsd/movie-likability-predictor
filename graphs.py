import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
import gensim.downloader as api

# -----------------
# Helper function to safely parse stringified lists
def safe_literal_eval(s):
    if pd.isna(s):
        return []
    try:
        return ast.literal_eval(str(s))
    except:
        return []

# -----------------
# Load the data
df = pd.read_csv('final_movie_reviews.csv')

# Apply safe parsing
df['movies_reviewed'] = df['movies_reviewed'].apply(safe_literal_eval)
df['ratings'] = df['ratings'].apply(safe_literal_eval)
df['overviews'] = df['overviews'].apply(safe_literal_eval)

# -----------------
# Load the Google News Word2Vec model
word_vectors = api.load("word2vec-google-news-300")  # <<<< ✅ loads the 300d Word2Vec model
print("✅ Model loaded.")

# -----------------
# Function to get average vector for an overview
def get_average_vector(text):
    words = simple_preprocess(text)  # tokenize
    vectors = [word_vectors[word] for word in words if word in word_vectors]
    if len(vectors) == 0:
        return np.zeros(word_vectors.vector_size)
    return np.mean(vectors, axis=0)

# -----------------
# Keep trying random users until one has enough data
found_valid_user = False

while not found_valid_user:
    sample_user = df.sample(1).iloc[0]

    liked_movies = []
    liked_overviews = []
    disliked_movies = []
    disliked_overviews = []

    for movie, rating, overview in zip(sample_user['movies_reviewed'], sample_user['ratings'], sample_user['overviews']):
        if isinstance(overview, str) and overview.strip():
            if rating == 1:
                liked_movies.append(movie)
                liked_overviews.append(overview)
            else:
                disliked_movies.append(movie)
                disliked_overviews.append(overview)

    # Only require 2 liked and 2 disliked
    liked_movies_sample = liked_movies[:2]
    liked_overviews_sample = liked_overviews[:2]
    recommended_movies_sample = disliked_movies[:2]
    recommended_overviews_sample = disliked_overviews[:2]

    liked_movies_clean = []
    liked_overviews_clean = []
    for m, o in zip(liked_movies_sample, liked_overviews_sample):
        if isinstance(o, str) and o.strip():
            liked_movies_clean.append(m)
            liked_overviews_clean.append(o)

    recommended_movies_clean = []
    recommended_overviews_clean = []
    for m, o in zip(recommended_movies_sample, recommended_overviews_sample):
        if isinstance(o, str) and o.strip():
            recommended_movies_clean.append(m)
            recommended_overviews_clean.append(o)

    if len(liked_overviews_clean) > 0 and len(recommended_overviews_clean) > 0:
        found_valid_user = True
        print(f"✅ Found a valid user: {sample_user['user_name']}")

# -----------------
# Now that we have a valid user:

# Encode overviews
liked_embeddings = np.array([get_average_vector(text) for text in liked_overviews_clean])
recommended_embeddings = np.array([get_average_vector(text) for text in recommended_overviews_clean])

# Calculate cosine similarity
similarity_matrix = cosine_similarity(liked_embeddings, recommended_embeddings)

# Create similarity DataFrame
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=liked_movies_clean,
    columns=recommended_movies_clean
)

# -----------------
# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(similarity_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title(f"Similarity Between Liked and Recommended Movies ({sample_user['user_name']})")
plt.xlabel("Recommended Movies")
plt.ylabel("Liked Movies")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Show the plot and keep it open
plt.show(block=True)
