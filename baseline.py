import pandas as pd
import numpy as np
import ast
import random
from gensim.downloader import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = load("glove-wiki-gigaword-50")
df = pd.read_csv('final_movie_reviews.csv')

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

random_accuracies = []
random_precisions = []
random_recalls = []
random_f1_scores = []

for idx, user_row in df.iterrows():
    movies = safe_literal_eval(user_row['movies_reviewed'])
    ratings = safe_literal_eval(user_row['ratings'])
    overviews = safe_literal_eval(user_row['overviews'])

    X = []
    y = []

    for overview, rating in zip(overviews, ratings):
        if isinstance(overview, str) and overview.strip():
            vector = get_avg_word2vec(overview, model)
            X.append(vector)
            y.append(rating)

    X = np.array(X)
    y = np.array(y)

    if len(X) < 2:
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(y_test) == 0:
        continue

    random_pred = [random.choice([0, 1]) for _ in range(len(y_test))]

    random_accuracies.append(accuracy_score(y_test, random_pred))
    random_precisions.append(precision_score(y_test, random_pred, zero_division=0))
    random_recalls.append(recall_score(y_test, random_pred, zero_division=0))
    random_f1_scores.append(f1_score(y_test, random_pred, zero_division=0))

print("Random Baseline Results (Averaged Across Users):")
print("Average Accuracy:", np.mean(random_accuracies))
print("Average Precision:", np.mean(random_precisions))
print("Average Recall:", np.mean(random_recalls))
print("Average F1 Score:", np.mean(random_f1_scores))
