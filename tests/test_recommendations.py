import unittest
import pandas as pd
from src.content_recommender import create_genre_matrix, create_similarity_matrix, recommend_similar_movies, recommend_popular_movies
from src.collaborative_recommender import recommend_collaborative_movies

class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.movies = pd.DataFrame({"MovieID": [1, 2, 3], "Title": ["A", "B", "C"], "Genres": ["Comedy", "Comedy", "Drama"], "avg_rating": [4., 5., 3.], "rating_count": [100, 1, 200]}, index=[10, 20, 30])
        self.sim = create_similarity_matrix(create_genre_matrix(self.movies))

    def test_tied_genres_exclude_selected_movie_and_handle_nondefault_index(self):
        recs = recommend_similar_movies("B", self.movies, self.sim, n=2)
        self.assertEqual(recs.MovieID.tolist(), [1, 3])

    def test_unknown_movie(self):
        self.assertTrue(recommend_similar_movies("Missing", self.movies, self.sim).empty)

    def test_duplicate_title_is_explicit(self):
        movies = self.movies.copy()
        movies.loc[20, "Title"] = "A"
        with self.assertRaises(ValueError):
            recommend_similar_movies("A", movies, self.sim)

    def test_popularity_excludes_low_support(self):
        recs = recommend_popular_movies(self.movies, min_ratings=100)
        self.assertNotIn(2, recs.MovieID.tolist())
        self.assertTrue(recs.weighted_rating.is_monotonic_decreasing)

    def test_collaborative_ties_exclude_selected_movie(self):
        movies = self.movies.rename(columns={"MovieID": "movie_id", "Title": "title", "Genres": "genres"})
        matrix = pd.DataFrame([[1, 1], [1, 1], [0, 1]], index=[1, 2, 3])
        from sklearn.metrics.pairwise import cosine_similarity
        recs = recommend_collaborative_movies(2, movies, matrix, cosine_similarity(matrix), n=2)
        self.assertEqual(recs.movie_id.tolist(), [1, 3])

    def test_zero_results(self):
        self.assertTrue(recommend_similar_movies("B", self.movies, self.sim, n=0).empty)

if __name__ == "__main__":
    unittest.main()
