"""
preprocess.py
--------------
Handles dataset acquisition and cleaning for the Moviefy project.

Responsibilities:
    1. Download the MovieLens ml-latest-small dataset if it is not
       already present in the `dataset/` folder.
    2. Load movies.csv and clean it using Pandas.
    3. Save the cleaned dataframe as movies.pkl for fast reuse.

Run this file directly to (re)generate movies.pkl:
    python preprocess.py
"""

import os
import zipfile
import urllib.request

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATASET_DIR = "dataset"
ZIP_PATH = os.path.join(DATASET_DIR, "ml-latest-small.zip")
EXTRACTED_FOLDER_NAME = "ml-latest-small"  # folder name inside the zip

MOVIES_CSV_PATH = os.path.join(DATASET_DIR, "movies.csv")
RATINGS_CSV_PATH = os.path.join(DATASET_DIR, "ratings.csv")

MOVIES_PKL_PATH = "movies.pkl"


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------
def download_dataset() -> None:
    """
    Download and extract the MovieLens ml-latest-small dataset if
    movies.csv and ratings.csv are not already present in dataset/.
    """
    if os.path.exists(MOVIES_CSV_PATH) and os.path.exists(RATINGS_CSV_PATH):
        print("Dataset already present. Skipping download.")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)

    print("Downloading MovieLens dataset...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    print("Download complete. Extracting...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    # The zip extracts into a subfolder named "ml-latest-small".
    # Move the two CSV files we need up into dataset/ directly.
    extracted_folder = os.path.join(DATASET_DIR, EXTRACTED_FOLDER_NAME)
    for filename in ("movies.csv", "ratings.csv"):
        src = os.path.join(extracted_folder, filename)
        dst = os.path.join(DATASET_DIR, filename)
        if os.path.exists(src):
            os.replace(src, dst)

    # Clean up the zip file and now-empty extracted folder.
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    if os.path.isdir(extracted_folder):
        try:
            os.rmdir(extracted_folder)
        except OSError:
            pass  # Folder not empty (extra files) -- safe to leave as is.

    print("Dataset ready in dataset/ folder.")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_movies_data() -> pd.DataFrame:
    """
    Load movies.csv, clean it, and return a tidy dataframe with
    only the columns needed for the recommendation engine.

    Cleaning steps:
        - Drop duplicate rows.
        - Drop rows with missing title/genres.
        - Keep only movieId, title, genres columns.
        - Remove movies with genres marked "(no genres listed)".
    """
    movies_df = pd.read_csv(MOVIES_CSV_PATH)

    # Keep only the columns we actually need.
    movies_df = movies_df[["movieId", "title", "genres"]]

    # Remove duplicate rows.
    movies_df = movies_df.drop_duplicates()

    # Handle missing values -- drop rows missing title or genres.
    movies_df = movies_df.dropna(subset=["title", "genres"])

    # Drop movies with no genre information; CountVectorizer needs
    # real genre text to build a meaningful similarity score.
    movies_df = movies_df[movies_df["genres"] != "(no genres listed)"]

    # Reset index after filtering so it's clean and sequential.
    movies_df = movies_df.reset_index(drop=True)

    return movies_df


def save_cleaned_data(movies_df: pd.DataFrame) -> None:
    """Save the cleaned dataframe as a pickle file for fast reuse."""
    movies_df.to_pickle(MOVIES_PKL_PATH)
    print(f"Cleaned dataset saved to {MOVIES_PKL_PATH} ({len(movies_df)} movies).")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_preprocessing() -> pd.DataFrame:
    """
    Full preprocessing pipeline: download (if needed), clean, and save.
    Returns the cleaned dataframe.
    """
    download_dataset()
    movies_df = clean_movies_data()
    save_cleaned_data(movies_df)
    return movies_df


if __name__ == "__main__":
    run_preprocessing()
