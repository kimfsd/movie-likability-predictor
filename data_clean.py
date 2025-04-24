import pandas as pd
import csv

def clean_reviews(ratings_df, films_df, max_reviews=100, max_users=1250):
    merged_df = pd.merge(ratings_df, films_df[['film_id', 'film_name']], on='film_id', how='left')
    
    df_limited = merged_df.groupby("user_name").head(max_reviews)
    top_users = df_limited["user_name"].unique()[:max_users]
    df_limited = df_limited[df_limited["user_name"].isin(top_users)]

    df_grouped = df_limited.groupby("user_name").agg({
        "film_name": lambda x: list(x)[:max_reviews],
        "rating": lambda x: list(x)[:max_reviews]
    }).reset_index()

    # in our new dataset, we are calling the tabs "movies_reviewed" and "ratings" respectively
    df_grouped = df_grouped.rename(columns={
        "film_name": "movies_reviewed",
        "rating": "ratings"
    })

    return df_grouped

def import_overviews(final_df, tmdb_csv_path):
    tmdb_data = pd.read_csv(tmdb_csv_path)
    
    # from what I saw, these were the column names used in the tmdb file
    possible_title_cols = ['original_title', 'title', 'film_name', 'name']
    title_col = next((col for col in possible_title_cols if col in tmdb_data.columns), None)

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