import pandas as pd
import numpy as np
import os
import json
import pickle
import random
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Paths and constants
CSV_FILE = "final_movie_reviews.csv"
EMBEDDING_FOLDER = "saved_embeddings"
W2V_MODEL_PATH = "GoogleNews-vectors-negative300.bin.gz"  # Adjust this if you have a different model
NUM_USERS_TO_ANALYZE = 500
MIN_RATINGS = 10

# Create folder for embeddings if it doesn't exist
os.makedirs(EMBEDDING_FOLDER, exist_ok=True)

def load_data():
    """Load and prepare the dataset"""
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} user records")
    return df

def parse_list_field(field_data):
    if isinstance(field_data, list):
        return field_data
    try:
        import ast
        return ast.literal_eval(field_data)
    except:
        try:
            return json.loads(field_data.replace("'", "\""))
        except:
            try:
                if field_data.startswith('[') and field_data.endswith(']'):
                    field_data = field_data[1:-1]
                return [item.strip().strip('"\'') for item in field_data.split(',')]
            except:
                print(f"Failed to parse: {field_data[:50]}...")
                return None

def extract_user_data(row):
    try:
        movies = parse_list_field(row["movies_reviewed"])
        ratings_raw = parse_list_field(row["ratings"])
        if ratings_raw:
            if isinstance(ratings_raw[0], str):
                ratings = [float(r.strip()) for r in ratings_raw]
            else:
                ratings = [float(r) for r in ratings_raw]
        else:
            ratings = None
        overviews = parse_list_field(row["overviews"])

        if not movies or not ratings or not overviews:
            return None
        if len(movies) != len(ratings) or len(ratings) != len(overviews):
            return None
        if len(movies) < MIN_RATINGS:
            return None
        if min(ratings) < 0 or max(ratings) > 5:
            return None

        return {
            "user": row["user_name"],
            "movies": movies,
            "ratings": ratings,
            "overviews": overviews,
        }
    except Exception as e:
        return None

def embed_and_cache(user_data, user_index, model):
    embed_path = f"{EMBEDDING_FOLDER}/user_{user_index}_embeddings.pkl"
    if os.path.exists(embed_path):
        try:
            with open(embed_path, "rb") as f:
                return pickle.load(f)
        except:
            pass

    def overview_to_vector(text):
        words = text.split()
        vectors = [model[word] for word in words if word in model]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    embeddings = np.array([overview_to_vector(overview) for overview in user_data["overviews"]])
    with open(embed_path, "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings

def find_best_threshold(embeddings, ratings):
    liked_indices = np.where(np.array(ratings) >= 4.0)[0]
    disliked_indices = np.where(np.array(ratings) <= 2.5)[0]

    liked = embeddings[liked_indices]
    disliked = embeddings[disliked_indices]

    if len(liked) < 2 or len(disliked) < 2:
        return None, None

    pos_scores = cosine_similarity(liked)[np.triu_indices(len(liked), k=1)] if len(liked) > 1 else np.array([])
    neg_scores = cosine_similarity(disliked)[np.triu_indices(len(disliked), k=1)] if len(disliked) > 1 else np.array([])

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None, None

    thresholds = np.arange(0.4, 0.95, 0.01)
    best_thresh, best_acc = 0, 0
    for t in thresholds:
        pos_preds = pos_scores >= t
        neg_preds = neg_scores < t
        acc = (np.sum(pos_preds) + np.sum(neg_preds)) / (len(pos_scores) + len(neg_scores))
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    return best_thresh, best_acc

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def predict_with_threshold(embeddings, ratings, threshold):
    liked_indices = np.where(np.array(ratings) >= 4.0)[0]
    if len(liked_indices) == 0:
        return np.random.randint(0, 2, size=len(ratings))

    mean_liked = np.mean(embeddings[liked_indices], axis=0)
    sims = cosine_similarity([mean_liked], embeddings)[0]
    preds = sims >= threshold
    return preds

def evaluate_random_baseline(ratings):
    y_true = np.array([1 if r >= 4.0 else 0 for r in ratings])
    positive_rate = np.mean(y_true)
    y_pred = np.random.random(size=len(y_true)) < positive_rate
    return calculate_metrics(y_true, y_pred)

def split_user_data(user_data, test_ratio=0.3):
    n = len(user_data['movies'])
    indices = np.arange(n)
    np.random.shuffle(indices)
    split_idx = int(n * (1 - test_ratio))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_data = {
        'user': user_data['user'],
        'movies': [user_data['movies'][i] for i in train_idx],
        'ratings': [user_data['ratings'][i] for i in train_idx],
        'overviews': [user_data['overviews'][i] for i in train_idx]
    }
    test_data = {
        'user': user_data['user'],
        'movies': [user_data['movies'][i] for i in test_idx],
        'ratings': [user_data['ratings'][i] for i in test_idx],
        'overviews': [user_data['overviews'][i] for i in test_idx]
    }
    return train_data, test_data

def main():
    df = load_data()

    # Add column for number of reviews
    df["num_reviews"] = df["movies_reviewed"].apply(lambda x: len(parse_list_field(x)) if pd.notna(x) else 0)
    df_sorted = df.sort_values(by="num_reviews", ascending=False)

    print("Loading Word2Vec model...")
    w2v_model = KeyedVectors.load_word2vec_format(W2V_MODEL_PATH, binary=True)

    users_data = []
    for i, row in df_sorted.iterrows():
        user_data = extract_user_data(row)
        if user_data:
            users_data.append(user_data)
        if len(users_data) >= NUM_USERS_TO_ANALYZE:
            break

    print(f"Analyzing {len(users_data)} users with most reviews")

    results = []

    for i, user_data in enumerate(tqdm(users_data, desc="Processing users")):
        train_data, test_data = split_user_data(user_data)
        train_embeddings = embed_and_cache(train_data, f"{i}_train", w2v_model)
        test_embeddings = embed_and_cache(test_data, f"{i}_test", w2v_model)

        threshold, train_acc = find_best_threshold(train_embeddings, train_data['ratings'])
        if threshold is None:
            continue

        test_preds = predict_with_threshold(test_embeddings, test_data['ratings'], threshold)
        test_labels = np.array([1 if r >= 4.0 else 0 for r in test_data['ratings']])

        threshold_metrics = calculate_metrics(test_labels, test_preds)
        baseline_metrics = evaluate_random_baseline(test_data['ratings'])

        user_result = {
            'user': user_data['user'],
            'threshold': threshold,
            'train_accuracy': train_acc,
            'threshold_metrics': threshold_metrics,
            'baseline_metrics': baseline_metrics,
            'improvement': threshold_metrics['accuracy'] - baseline_metrics['accuracy']
        }

        results.append(user_result)

        print(f"\n[User {i}: {user_data['user']}]")
        print(f"Threshold: {threshold:.2f}, Train Acc: {train_acc:.2f}")
        print(f"Test Metrics: {threshold_metrics}")
        print(f"Baseline Metrics: {baseline_metrics}")
        print(f"Accuracy Gain: {user_result['improvement']:.2f}")

    if results:
        print("\n===== OVERALL RESULTS =====")
        avg_threshold = np.mean([r['threshold'] for r in results])
        avg_improvement = np.mean([r['improvement'] for r in results])
        threshold_accuracies = [r['threshold_metrics']['accuracy'] for r in results]
        baseline_accuracies = [r['baseline_metrics']['accuracy'] for r in results]

        print(f"Avg Threshold: {avg_threshold:.2f}")
        print(f"Avg Threshold Acc: {np.mean(threshold_accuracies):.2f}")
        print(f"Avg Baseline Acc: {np.mean(baseline_accuracies):.2f}")
        print(f"Avg Improvement: {avg_improvement:.2f}")
        print(f"Users Improved: {sum(r['improvement'] > 0 for r in results)}/{len(results)}")

if __name__ == "__main__":
    main()
