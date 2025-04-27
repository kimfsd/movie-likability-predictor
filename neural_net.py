import pandas as pd
import numpy as np
import ast
from gensim.downloader import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

print("Loading GloVe embeddings...")
model = load("glove-wiki-gigaword-300")

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

def prepare_data(df):
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

def build_and_train_nn(X_train, y_train, X_test, y_test, label=""):
    nn = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    nn.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)
    
    y_pred_prob = nn.predict(X_test).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print(f"\nNeural Network Results ({label}):")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

X, y = prepare_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
build_and_train_nn(X_train, y_train, X_test, y_test, label="With Stopwords")

# Prepare stopwords-removed data
X_no_stop, y_no_stop = prepare_data(df_no_stop)
X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(X_no_stop, y_no_stop, test_size=0.2, random_state=42)
build_and_train_nn(X_train_ns, y_train_ns, X_test_ns, y_test_ns, label="Without Stopwords")
