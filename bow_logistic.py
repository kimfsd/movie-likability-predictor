# all for bag of words logistic regression

import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

accuracies = []
precisions = []
recalls = []
f1_scores = []

for idx, user_row in df.iterrows():
    movies = safe_literal_eval(user_row['movies_reviewed'])
    ratings = safe_literal_eval(user_row['ratings'])
    overviews = safe_literal_eval(user_row['overviews'])

    X_texts = []
    y = []

    for overview, rating in zip(overviews, ratings):
        if isinstance(overview, str) and overview.strip():
            X_texts.append(overview)
            y.append(rating)

    if len(X_texts) < 2:
        continue

    # Bag of Words vectorization
    vectorizer = CountVectorizer(min_df=2, max_df=0.8, stop_words='english')
    X = vectorizer.fit_transform(X_texts)

    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(np.unique(y_train)) < 2:
        print(f"Skipping user {idx}: only one class present.")
        continue

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, zero_division=0))
    recalls.append(recall_score(y_test, y_pred, zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

print("Average Accuracy:", np.mean(accuracies))
print("Average Precision:", np.mean(precisions))
print("Average Recall:", np.mean(recalls))
print("Average F1 Score:", np.mean(f1_scores))
