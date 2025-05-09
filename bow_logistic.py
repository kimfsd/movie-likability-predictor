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

# Track per-user metrics
user_metrics = []

for idx, user_row in df.iterrows():
    user_name = user_row['user_name']
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if len(np.unique(y_train)) < 2:
        print(f"Skipping user {user_name}: only one class present.")
        continue

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    user_metrics.append((user_name, acc, prec, rec, f1))

# After the loop
accuracies = [x[1] for x in user_metrics]
precisions = [x[2] for x in user_metrics]
recalls = [x[3] for x in user_metrics]
f1_scores = [x[4] for x in user_metrics]

print("\n--- Overall Metrics ---")
print("Average Accuracy:", np.mean(accuracies))
print("Average Precision:", np.mean(precisions))
print("Average Recall:", np.mean(recalls))
print("Average F1 Score:", np.mean(f1_scores))

# Find the user with the highest accuracy
best_user = max(user_metrics, key=lambda x: x[1])

print("\n--- Best Performing User ---")
print(f"User: {best_user[0]}")
print(f"Accuracy: {best_user[1]:.4f}")
print(f"Precision: {best_user[2]:.4f}")
print(f"Recall: {best_user[3]:.4f}")
print(f"F1 Score: {best_user[4]:.4f}")
