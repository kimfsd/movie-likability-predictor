import pandas as pd
import csv, re, ast
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_reviews(ratings_df, films_df, max_reviews=100, max_users=1250):
    merged_df = pd.merge(ratings_df, films_df[['film_id', 'film_name']], on='film_id', how='left')
    
    df_limited = merged_df.groupby("user_name").head(max_reviews)
    top_users = df_limited["user_name"].unique()[:max_users]
    df_limited = df_limited[df_limited["user_name"].isin(top_users)]

    df_grouped = df_limited.groupby("user_name").agg({
        "film_name": lambda x: list(x)[:max_reviews],
        "rating": lambda x: list(x)[:max_reviews]
    }).reset_index()

    df_grouped = df_grouped.rename(columns={
        "film_name": "movies_reviewed",
        "rating": "ratings"
    })
    
    return df_grouped
    
def delete_no_overview(final_df, unmatched_csv_path):
    unmatched_df = pd.read_csv(unmatched_csv_path)
    unmatched_titles = set(unmatched_df.iloc[:, 0].dropna().astype(str).str.strip().str.lower())

    def filter_user_row(row):
        filtered = [
            (movie, rating, overview)
            for movie, rating, overview in zip(row["movies_reviewed"], row["ratings"], row["overviews"])
            if isinstance(movie, str) and movie.strip().lower() not in unmatched_titles
        ]
        if filtered:
            movies, ratings, overviews = zip(*filtered)
            return pd.Series([list(movies), list(ratings), list(overviews)])
        else:
            return pd.Series([[], [], []])

    # Clean unmatched titles
    final_df[["movies_reviewed", "ratings", "overviews"]] = final_df.apply(filter_user_row, axis=1)

    # Binarize ratings
    final_df["ratings"] = final_df["ratings"].apply(lambda ratings: [1 if isinstance(r, (int, float)) and r >= 4.0 else 0 for r in ratings])

    # Step 1: Keep only users with at least 10 movies
    final_df = final_df[final_df["movies_reviewed"].apply(len) >= 10].reset_index(drop=True)

    # Step 2: Keep only users with at least 2 reviews for both 0s and 1s
    def has_minimum_class_counts(ratings, min_per_class=2):
        counts = pd.Series(ratings).value_counts()
        return counts.get(0, 0) >= min_per_class and counts.get(1, 0) >= min_per_class

    final_df = final_df[final_df["ratings"].apply(has_minimum_class_counts)].reset_index(drop=True)

    return final_df

    return df_grouped

def import_overviews(final_df, tmdb_csv_path):
    tmdb_data = pd.read_csv(tmdb_csv_path)
    
    title_cols = ['original_title', 'title', 'film_name', 'name']
    title_col = next((col for col in title_cols if col in tmdb_data.columns), None)

    if not title_col or 'overview' not in tmdb_data.columns:
        raise ValueError("Error check: This file doesn't have an overview or title column")

    tmdb_map = dict(zip(tmdb_data[title_col].str.lower(), tmdb_data['overview']))

    def get_overviews(movie_list):
        return [
            tmdb_map.get(title.lower(), "Overview doesn't exist")
            if isinstance(title, str) else "Overview doesn't exist"
            for title in movie_list
        ]

    final_df["overviews"] = final_df["movies_reviewed"].apply(get_overviews)
    return final_df

def remove_courtesy_tag(final_df):
    final_df["overviews"] = final_df["overviews"].apply(
        lambda lst: [
            re.sub(r"\s*\[Overview Courtesy of Gladstone Gallery\]\s*", "", o).strip() if isinstance(o, str) else o
            for o in lst
        ]
    )
    return final_df

def no_overview_list(final_df, output_path="unmatched_titles.csv"):
    unmatched = set()

    for titles, overviews in zip(final_df["movies_reviewed"], final_df["overviews"]):
        for title, overview in zip(titles, overviews):
            if overview == "Overview doesn't exist" and isinstance(title, str):
                unmatched.add(title)

    with open(output_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Movie titles without overview"])
        for title in sorted(unmatched):
            writer.writerow([title])

    print(f"Output {len(unmatched)} movies without overview to: '{output_path}'")

def delete_no_overview(final_df, unmatched_csv_path):
    unmatched_df = pd.read_csv(unmatched_csv_path)
    unmatched_titles = set(unmatched_df.iloc[:, 0].dropna().astype(str).str.strip().str.lower())

    def filter_user_row(row):
        filtered = [
            (movie, rating, overview)
            for movie, rating, overview in zip(row["movies_reviewed"], row["ratings"], row["overviews"])
            if isinstance(movie, str) and movie.strip().lower() not in unmatched_titles
        ]
        if filtered:
            movies, ratings, overviews = zip(*filtered)
            return pd.Series([list(movies), list(ratings), list(overviews)])
        else:
            return pd.Series([[], [], []])

    final_df[["movies_reviewed", "ratings", "overviews"]] = final_df.apply(filter_user_row, axis=1)
    final_df["ratings"] = final_df["ratings"].apply(lambda ratings: [1 if isinstance(r, (int, float)) and r >= 4.0 else 0 for r in ratings])

    final_df = final_df[final_df["movies_reviewed"].apply(len) >= 10].reset_index(drop=True)

    def has_minimum_class_counts(ratings, min_per_class=2):
        counts = pd.Series(ratings).value_counts()
        return counts.get(0, 0) >= min_per_class and counts.get(1, 0) >= min_per_class

    final_df = final_df[final_df["ratings"].apply(has_minimum_class_counts)].reset_index(drop=True)

    return final_df


def remove_stopwords_and_lowercase(final_df, output_path="final_removed_stop.csv"):
    def clean_overview_list(overview_list):
        cleaned_list = []
        for overview in overview_list:
            if isinstance(overview, str):
                words = overview.lower().split()
                filtered_words = [word for word in words if word not in stop_words]
                cleaned_overview = " ".join(filtered_words)
                cleaned_list.append(cleaned_overview)
            else:
                cleaned_list.append(overview)
        return cleaned_list

    final_df_copy = final_df.copy()
    final_df_copy["overviews"] = final_df_copy["overviews"].apply(clean_overview_list)
    final_df_copy.to_csv(output_path, index=False)
    print(f"Output successful in: '{output_path}'")

    return final_df_copy

ratings_df = pd.read_csv("data/ratings.csv")
films_df = pd.read_csv("data/films.csv", on_bad_lines='skip', engine='python')

final_df = clean_reviews(ratings_df, films_df)
final_df = import_overviews(final_df, "data/TMDB_movie_dataset_v11.csv")
final_df = remove_courtesy_tag(final_df)

no_overview_list(final_df, "unmatched_titles.csv")
final_df = delete_no_overview(final_df, "unmatched_titles.csv")

final_df.to_csv("final_movie_reviews.csv", index=False)
print("Output successful in: 'final_movie_reviews.csv'")

remove_stopwords_and_lowercase(final_df, "final_removed_stop.csv")