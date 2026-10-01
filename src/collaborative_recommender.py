import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def create_user_movie_matrix(ratings):
    """
    Create a user-movie rating matrix.
    """
    return ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    )


def create_movie_user_matrix(user_movie_matrix):
    """
    Transpose the user-movie matrix so that
    each row represents a movie.
    """
    return user_movie_matrix.T


def create_collaborative_similarity(movie_user_matrix):
    """
    Calculate item-item cosine similarity
    using movie rating patterns.
    """

    movie_user_matrix_filled = movie_user_matrix.fillna(0)

    return cosine_similarity(
        movie_user_matrix_filled
    )


def recommend_collaborative_movies(
    movie_id,
    movies,
    movie_user_matrix,
    cosine_sim_cf,
    n=10
):
    """
    Recommend movies using item-item
    collaborative filtering.
    """

    if movie_id not in movie_user_matrix.index:
        return pd.DataFrame()

    # Find the row position of the movie
    idx = movie_user_matrix.index.get_loc(movie_id)

    # Get similarity scores
    similarity_scores = list(
        enumerate(cosine_sim_cf[idx])
    )

    # Sort from most similar to least similar
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the movie itself
    similarity_scores = [(i, score) for i, score in similarity_scores if i != idx][:max(0, n)]

    recommendations = []

    for movie_index, score in similarity_scores:

        recommended_movie_id = (
            movie_user_matrix.index[movie_index]
        )

        movie_info = movies[
            movies["movie_id"] == recommended_movie_id
        ]

        if not movie_info.empty:

            recommendations.append({
                "movie_id": recommended_movie_id,
                "title": movie_info.iloc[0]["title"],
                "genres": movie_info.iloc[0]["genres"],
                "similarity": score
            })

    return pd.DataFrame(recommendations)