import pandas as pd
import numpy as np
import ast
from gensim.downloader import load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = load("glove-wiki-gigaword-50")

df = pd.read_csv('final_movie_reviews.csv')
df_no_stop = pd.read_csv('final_removed_stop.csv')

def safe_literal_eval(s):
    if pd.isna(s):
        return []
    s = str(s).replace('nan', 'None')
    try:
        return ast.literal_eval(s)
    except Exception as e:
        print(f"Failed to parse: {s}")
        raise e

def get_avg_word2vec(overview, model):
    if isinstance(overview, str):
        words = overview.split()
        vectors = [model[word] for word in words if word in model]
        if vectors:
            return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

def prepare_X_y(df, model):
    X = []
    y = []
    for idx, user_df in df.iterrows():
        movies = safe_literal_eval(user_df['movies_reviewed'])
        ratings = safe_literal_eval(user_df['ratings'])
        overviews = safe_literal_eval(user_df['overviews'])

        for overview, rating in zip(overviews, ratings):
            if isinstance(overview, str) and overview.strip():
                vector = get_avg_word2vec(overview, model)
                X.append(vector)
                y.append(rating)
    return np.array(X), np.array(y)

def evaluate_knn(X, y, name="Dataset"):
    if len(X) < 2:
        print(f"{name}: Not enough data to train/test split.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(f"\nResults for {name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    if len(X) >= 2:
        sim_score = cosine_similarity([X[0]], [X[1]])[0][0]
        print(f"Cosine similarity between first two movie overviews: {sim_score:.4f}")


X_orig, y_orig = prepare_X_y(df, model)
evaluate_knn(X_orig, y_orig, name="Original (with stopwords)")

X_clean, y_clean = prepare_X_y(df_no_stop, model)
evaluate_knn(X_clean, y_clean, name="Stopwords Removed (cleaned)")