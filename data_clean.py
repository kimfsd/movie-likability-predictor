import pandas as pd
import csv, re

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

# to remove the Gladstone Gallery tag

""" might have to add more because I saw some other tags (possibly) """

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
        write = csv.writer(f)
        write.writerow(["Movie titles without overview"])
        for title in sorted(unmatched):
            write.writerow([title])

    print(f"Output {len(unmatched)} movies without overview to: '{output_path}'")
    
def delete_no_overview(final_output, unmatched_csv_path):
    # Load and normalize unmatched titles
    unmatched_df = pd.read_csv(unmatched_csv_path)
    unmatched_titles = set(unmatched_df.iloc[:, 0].dropna().astype(str).str.strip().str.lower())

    def filter_user_row(row):
        # we filter through each of the movies in the MAIN csv
        # if the movie title is in the unmatched list, then remove the ratings, the movie title, and the overviews
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

    final_output[["movies_reviewed", "ratings", "overviews"]] = final_output.apply(filter_user_row, axis=1)
    final_output = final_output[final_output["movies_reviewed"].apply(len) > 0].reset_index(drop=True)

    return final_output

ratings_df = pd.read_csv("data/ratings.csv")
films_df = pd.read_csv("data/films.csv", on_bad_lines='skip', engine='python')

# first call that gets the ratings, movies, and user_name(s)
final_df = clean_reviews(ratings_df, films_df)

# this gets us the overviews! check this function if there is an error!
final_df = import_overviews(final_df, "data/TMDB_movie_dataset_v11.csv")
final_df = remove_courtesy_tag(final_df)

# we export the list with no overviews and remove them here
no_overview_list(final_df, "unmatched_titles.csv")
final_df = delete_no_overview(final_df, "unmatched_titles.csv")


final_df.to_csv("final_movie_reviews.csv", index=False)
print("Output successful in: 'final_movie_reviews.csv'")