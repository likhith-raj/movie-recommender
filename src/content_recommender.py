import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def create_genre_matrix(movie_stats):
    """
    Convert movie genres into a multi-hot encoded matrix.
    """

    return movie_stats["Genres"].str.get_dummies(sep="|")


def create_similarity_matrix(genre_matrix):
    """
    Calculate cosine similarity between all movies.
    """

    return cosine_similarity(genre_matrix)


def recommend_popular_movies(
    movie_stats,
    n=10,
    min_ratings=100
):
    """
    Recommend popular movies using weighted ratings.
    """

    C = movie_stats["avg_rating"].mean()

    qualified_movies = movie_stats[
        movie_stats["rating_count"] >= min_ratings
    ].copy()

    qualified_movies["weighted_rating"] = (
        (qualified_movies["rating_count"] /
         (qualified_movies["rating_count"] + min_ratings))
        * qualified_movies["avg_rating"]
        +
        (min_ratings /
         (qualified_movies["rating_count"] + min_ratings))
        * C
    )

    recommendations = (
        qualified_movies
        .sort_values(
            "weighted_rating",
            ascending=False
        )
        .head(n)
    )

    return recommendations[
        [
            "MovieID",
            "Title",
            "Genres",
            "avg_rating",
            "rating_count",
            "weighted_rating"
        ]
    ]


def recommend_similar_movies(
    movie_title,
    movie_stats,
    cosine_sim,
    n=10
):
    """
    Recommend movies similar to a selected movie
    based on genre similarity.
    """

    positions = [i for i, title in enumerate(movie_stats["Title"]) if title == movie_title]
    if not positions:
        return pd.DataFrame(columns=["MovieID", "Title", "Genres", "similarity"])
    if len(positions) > 1:
        raise ValueError("Movie title is ambiguous; select a unique title.")
    idx = positions[0]

    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the selected movie itself
    similarity_scores = [(i, score) for i, score in similarity_scores if i != idx][:max(0, n)]

    recommendations = []

    for movie_index, score in similarity_scores:

        recommendations.append({
            "MovieID": movie_stats.iloc[movie_index]["MovieID"],
            "Title": movie_stats.iloc[movie_index]["Title"],
            "Genres": movie_stats.iloc[movie_index]["Genres"],
            "similarity": score
        })

    return pd.DataFrame(recommendations)