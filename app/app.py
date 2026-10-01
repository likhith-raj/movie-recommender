import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to Python path
PROJECT_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_PATH))

# Import our recommendation functions
from src.content_recommender import (
    create_genre_matrix,
    create_similarity_matrix,
    recommend_similar_movies,
    recommend_popular_movies
)


# -----------------------------
# Load data
# -----------------------------

PROCESSED_PATH = PROJECT_PATH / "data" / "processed"

@st.cache_data
def load_data():
    return pd.read_csv(PROCESSED_PATH / "movie_stats.csv")


@st.cache_resource
def build_similarity(movie_stats):
    return create_similarity_matrix(create_genre_matrix(movie_stats))


if not (PROCESSED_PATH / "movie_stats.csv").exists():
    st.info("Movie data is missing. Follow the data setup steps in the README.")
    st.stop()

movie_stats = load_data()
cosine_sim = build_similarity(movie_stats)


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🎬 Movie Recommender")

st.write(
    "Find movies similar to your favourite movie "
    "using genre-based recommendations."
)


# -----------------------------
# Similar Movies
# -----------------------------

st.header("🔎 Find Similar Movies")

movie_title = st.selectbox(
    "Select a movie:",
    movie_stats["Title"].sort_values()
)

if st.button("Recommend Movies"):

    recommendations = recommend_similar_movies(
        movie_title,
        movie_stats,
        cosine_sim,
        n=10
    )

    st.subheader(
        f"Movies similar to **{movie_title}**"
    )

    st.dataframe(
        recommendations,
        use_container_width=True
    )


# -----------------------------
# Popular Movies
# -----------------------------

st.header("🔥 Popular Movies")

popular_movies = recommend_popular_movies(
    movie_stats,
    n=10,
    min_ratings=100
)

st.dataframe(
    popular_movies,
    use_container_width=True
)