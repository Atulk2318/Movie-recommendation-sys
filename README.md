# 🎬 Moviefy

A content-based movie recommendation system built with **Streamlit**,
using genre similarity (CountVectorizer + Cosine Similarity) on the
**MovieLens ml-latest-small** dataset. Enriched with poster, rating,
and plot details from the **OMDb API**.

Built as a B.Tech student project — simple, modular, and readable.

---

## Features

- Content-based recommendations using genre similarity
- Choose Top 5 or Top 10 recommendations
- Movie posters, IMDb ratings, plot, cast, and director via OMDb API
- Cached pickle files (`movies.pkl`, `similarity.pkl`) for fast reloads
- Clean dark-themed, responsive UI
- Graceful error handling — never crashes on missing data or API errors

---

## Project Structure

```text
Moviefy/
│
├── app.py                 # Streamlit UI and main entry point
├── preprocess.py          # Dataset download + cleaning
├── recommendation.py      # CountVectorizer + Cosine Similarity engine
├── requirements.txt
├── movies.pkl              # generated on first run
├── similarity.pkl          # generated on first run
│
├── dataset/                # MovieLens CSVs (auto-downloaded)
├── css/
│   └── style.css
├── assets/
│   └── logo.png
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

---

## Setup (Local)

1. **Clone the project and install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OMDb API key** (get a free key at
   https://www.omdbapi.com/apikey.aspx):

   ```bash
   export OMDB_API_KEY=your_key_here      # macOS/Linux
   set OMDB_API_KEY=your_key_here         # Windows (cmd)
   ```

   Or copy `.env.example` to `.env` and load it with a tool like
   `python-dotenv`, or use Streamlit secrets (see below).

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

   On first run, `app.py` automatically downloads the MovieLens
   dataset, cleans it, builds the similarity matrix, and caches both
   as `movies.pkl` and `similarity.pkl`. Subsequent runs load the
   cached files instantly.

---

## Deploying to Streamlit Community Cloud

1. Push this project to a GitHub repository (the `.gitignore`
   already excludes generated pickle files and datasets — they'll
   be rebuilt automatically on first load).
2. Go to [share.streamlit.io](https://share.streamlit.io) and create
   a new app pointing to `app.py`.
3. In the app's **Settings → Secrets**, paste:

   ```toml
   OMDB_API_KEY = "your_omdb_api_key_here"
   ```

   (see `.streamlit/secrets.toml.example` for reference)

4. Deploy. The app installs `requirements.txt`, downloads the
   dataset, and builds its cache automatically.

---

## How It Works

1. **`preprocess.py`** downloads `ml-latest-small.zip`, extracts
   `movies.csv` and `ratings.csv`, cleans the movies data (removes
   duplicates, missing values, and genre-less entries), and saves
   the result as `movies.pkl`.

2. **`recommendation.py`** converts each movie's `genres` string
   (e.g. `"Adventure|Animation|Comedy"`) into a count vector, then
   computes pairwise cosine similarity across all movies. The result
   is cached as `similarity.pkl`. `get_recommendations(title, n)`
   returns the `n` most similar movies to a given title.

3. **`app.py`** ties it together: a Streamlit UI lets the user pick
   a movie and a recommendation count, then displays the results as
   cards enriched with live data from the OMDb API.

---

## Notes

- The OMDb API key is never hardcoded — it's read from
  `st.secrets` (Streamlit Cloud) or the `OMDB_API_KEY` environment
  variable, with a fallback demo key so the app still runs out of
  the box for local testing.
- If OMDb is unreachable or a movie isn't found, the app shows a
  placeholder poster and "N/A" for missing fields instead of
  crashing.
