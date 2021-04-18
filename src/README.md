# Data Generation
While some data is included in the data/ folder, some are not (such as the datasets with comment data for sentiment analysis).
To generate all the data, [download the dataset from Kaggle](https://www.kaggle.com/dahlia25/metacritic-video-game-comments) and
place the .csv files into the data/ folder and then run the `reformat_and_split_data.ipynb` notebook.

# Content-Based Filtering

All content-based filtering code is contained in `content_based_filtering.ipynb`. 

The results of the permutation test scores are contained in `content_permutation_test_scores.csv`

# Collaborative Filtering

## K-Nearest Neighbors

All code related to the K-Nearest Neighbors approach is contained in the files `nearest_neighbor.py` and `nearest_neighbor_analysis.py`

## Embedding Matrices

The final code for to embedding matrices is contained in the `embeddings3.ipynb` notebook.

## Deep Matrix Factorization

All code related to deep matrix factorization can be found in the notebook `Deep Matrix Factorization.ipynb`.

# Sentiment Analysis

All code related to sentiment analysis are contained in files prefixed with `sentiment_analysis`. The trained models are contained in the `models/` folder.
