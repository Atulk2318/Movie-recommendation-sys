"""
recommendation.py
------------------
Content-based movie recommendation engine for Moviefy.

Builds a genre-based similarity matrix using CountVectorizer +
Cosine Similarity, and exposes a simple get_recommendations()
function used by the Streamlit app.

Run this file directly to (re)generate similarity.pkl:
    python recommendation.py
"""

import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import run_preprocessing, MOVIES_PKL_PATH

SIMILARITY_PKL_PATH = "similarity.pkl"


# ---------------------------------------------------------------------------
# Loading / building the movies dataframe
# ---------------------------------------------------------------------------
def load_movies() -> pd.DataFrame:
    """
    Load the cleaned movies dataframe from movies.pkl.
    If it doesn't exist yet, run the preprocessing pipeline first.
    """
    if not os.path.exists(MOVIES_PKL_PATH):
        return run_preprocessing()
    return pd.read_pickle(MOVIES_PKL_PATH)


# ---------------------------------------------------------------------------
# Building the similarity matrix
# ---------------------------------------------------------------------------
def build_similarity_matrix(movies_df: pd.DataFrame):
    """
    Convert the genres column into count vectors and compute the
    cosine similarity between every pair of movies.

    Genres in the dataset look like "Adventure|Animation|Comedy".
    We replace the '|' separator with a space so CountVectorizer
    treats each genre as its own token/word.
    """
    genre_text = movies_df["genres"].str.replace("|", " ", regex=False)

    vectorizer = CountVectorizer()
    genre_vectors = vectorizer.fit_transform(genre_text)

    similarity_matrix = cosine_similarity(genre_vectors)
    return similarity_matrix


def save_similarity_matrix(similarity_matrix) -> None:
    """Save the similarity matrix as a pickle file for fast reuse."""
    pd.to_pickle(similarity_matrix, SIMILARITY_PKL_PATH)
    print(f"Similarity matrix saved to {SIMILARITY_PKL_PATH}.")


def load_or_build_similarity_matrix(movies_df: pd.DataFrame):
    """
    Load the similarity matrix from similarity.pkl if it exists,
    otherwise build it fresh and cache it to disk.
    """
    if os.path.exists(SIMILARITY_PKL_PATH):
        return pd.read_pickle(SIMILARITY_PKL_PATH)

    similarity_matrix = build_similarity_matrix(movies_df)
    save_similarity_matrix(similarity_matrix)
    return similarity_matrix


# ---------------------------------------------------------------------------
# Public recommendation function
# ---------------------------------------------------------------------------
def get_recommendations(movie_title: str, n: int = 5, movies_df=None, similarity_matrix=None):
    """
    Return the top-n movies most similar to the given movie title,
    based on genre similarity.

    Parameters
    ----------
    movie_title : str
        The exact title of the movie to base recommendations on
        (must match a title in movies.pkl).
    n : int
        Number of recommendations to return. Expected to be 5 or 10.
    movies_df : pd.DataFrame, optional
        Preloaded movies dataframe. If not provided, it is loaded
        from disk (slower -- prefer passing it in from the app).
    similarity_matrix : np.ndarray, optional
        Preloaded similarity matrix. If not provided, it is loaded
        or built from disk.

    Returns
    -------
    pd.DataFrame
        A dataframe of the top-n recommended movies with an added
        'similarity_score' column, sorted by similarity descending.
        Returns an empty dataframe if the title is not found.
    """
    if movies_df is None:
        movies_df = load_movies()
    if similarity_matrix is None:
        similarity_matrix = load_or_build_similarity_matrix(movies_df)

    # Find the index of the requested movie.
    matches = movies_df.index[movies_df["title"] == movie_title].tolist()
    if not matches:
        return pd.DataFrame(columns=["movieId", "title", "genres", "similarity_score"])

    movie_index = matches[0]

    # Get similarity scores for this movie against all others.
    scores = list(enumerate(similarity_matrix[movie_index]))

    # Sort by similarity score, descending. Skip index 0 result
    # since that will always be the movie itself (score == 1.0).
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    scores = [item for item in scores if item[0] != movie_index][:n]

    recommended_indices = [item[0] for item in scores]
    similarity_scores = [item[1] for item in scores]

    recommendations = movies_df.iloc[recommended_indices].copy()
    recommendations["similarity_score"] = similarity_scores
    recommendations = recommendations.sort_values("similarity_score", ascending=False)
    recommendations = recommendations.reset_index(drop=True)

    return recommendations


if __name__ == "__main__":
    # Quick manual test when running this file directly.
    movies = load_movies()
    sim_matrix = load_or_build_similarity_matrix(movies)

    sample_title = movies["title"].iloc[0]
    print(f"Sample recommendations for: {sample_title}\n")
    print(get_recommendations(sample_title, n=5, movies_df=movies, similarity_matrix=sim_matrix))
