import pandas as pd
import ast
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
df = pd.read_csv('final_movie_reviews.csv')

def safe_literal_eval(s):
    if pd.isna(s):
        return []
    s = str(s).replace('nan', 'None')
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

token_counts = []

for idx, user_df in df.iterrows():
    overviews = safe_literal_eval(user_df['overviews'])
    
    for overview in overviews:
        if isinstance(overview, str) and overview.strip() and "Overview doesn't exist" not in overview:
            tokens = word_tokenize(overview)
            token_counts.append(len(tokens))

if token_counts:
    avg_tokens_per_overview = sum(token_counts) / len(token_counts)
    print(f"Average number of tokens per overview: {avg_tokens_per_overview:.2f}")
    print(f"Total number of overviews counted: {len(token_counts)}")
else:
    print("No valid overviews found.")
