# movie-likability-predictor

**_Structure:_**

- The "data" folder hosts the original csv files that we modified in order to get our "final_movie_reviews.csv"
- "intermediate_data" just holds the files generated when we removed the unwanted movies in our dataset, because of not having a synopsis or niche titles.
- All models are just in the main directory in each of the .py files.

**_Data Files:_**

- The main data file is "final_movie_reviews.csv" which is generated from the "data" directory and some other modified csv files that we generated ourselves

**_Running Code:_**

- you could choose the model you want to run by simply doing: python3 (or python) [filename.py]
- ex) python3 neural_net.py

**_Datasets to Download (if you want to start fresh locally):_**

Letterbox Ratings: https://www.kaggle.com/datasets/freeth/letterboxd-film-ratings?select=ratings.csv

- ! You only need these two csv: **films.csv and ratings.csv**

Full TMDB Movies: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies

- more specifically, you need to have "TMDB_movie_dataset_v11.csv"

Once all datasets are downloaded, put all the csv files into a folder called "data" that you create. Paste all the csv files into the "data" folder and run:
python3 (or python) data_clean.py to get the "final_movie_reviews.csv"

Don't worry too much about the unmatched_titles.csv... it was an intermediate step to remove some unwanted movies
