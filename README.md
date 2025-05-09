# movie-likability-predictor

Directory Structure:

Data Files:
-- The main data file is "final_movie_reviews.csv" which is generated from the "data" directory and some other modified csv files that we generated ourselves

Running Code:
-- you could choose the model you want to run by simply doing: python3 (or python) [filename.py]
-- ex) python3 neural_net.py

Datasets to Download (if you want to start fresh locally):
Letterbox Ratings: https://www.kaggle.com/datasets/freeth/letterboxd-film-ratings?select=ratings.csv
-- you need to have the films.csv and ratings.csv!!!
Full TMDB Movies: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
-- more specifically, you need to have "TMDB_movie_dataset_v11.csv"

Once all datasets are downloaded, put all the csv files into a folder called "data" that you create. Paste all the csv files into the "data" folder and run:
python3 (or python) data_clean.py to get the "final_movie_reviews.csv"

Don't worry too much about the unmatched_titles.csv... it was an intermediate step to remove some unwanted movies
