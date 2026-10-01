"""Frozen baseline comparison with a global chronological holdout.

Run from the repository root. No test-derived tuning; no raw user-level output.
"""
from pathlib import Path
import argparse
import json
import hashlib
import sys
import platform
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.content_recommender import create_genre_matrix, create_similarity_matrix, recommend_popular_movies
from src.collaborative_recommender import create_user_movie_matrix, create_movie_user_matrix, create_collaborative_similarity


def temporal_split(ratings, fraction=0.8):
    if not 0 < fraction < 1:
        raise ValueError("Train fraction must be between zero and one.")
    timestamps = ratings.timestamp.sort_values().to_numpy()
    cutoff = int(timestamps[int(len(timestamps) * fraction)])
    # Keep every equal-timestamp event on the same side of the boundary.
    train = ratings[ratings.timestamp < cutoff].copy()
    test = ratings[ratings.timestamp >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split needs distinct timestamps on both sides.")
    return train, test, cutoff


def ranking_metrics(recommendations, relevant, k=10):
    ranked = list(recommendations)[:k]
    if len(ranked) != len(set(ranked)):
        raise ValueError("Ranked list contains duplicates.")
    relevant = set(relevant)
    if not relevant:
        raise ValueError("At least one relevant item is required.")
    hits = np.array([movie in relevant for movie in ranked], dtype=float)
    recall = float(hits.sum() / len(relevant))
    precision = float(hits.sum() / k)
    discounts = 1 / np.log2(np.arange(2, len(ranked) + 2))
    ideal = float((1 / np.log2(np.arange(2, min(k, len(relevant)) + 2))).sum())
    return recall, float((hits * discounts).sum() / ideal), precision


def evaluate(raw_dir, output_dir):
    ratings_path = raw_dir / "ratings.dat"
    ratings = pd.read_csv(ratings_path, sep="::", engine="python",
                          names=["user_id", "movie_id", "rating", "timestamp"])
    movies = pd.read_csv(raw_dir / "movies.dat", sep="::", engine="python",
                         names=["movie_id", "title", "genres"], encoding="latin-1")
    if ratings.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("Duplicate user/movie ratings need an explicit resolution policy.")
    train, test, cutoff = temporal_split(ratings)
    stats = train.groupby("movie_id").rating.agg(avg_rating="mean", rating_count="count")
    catalogue = movies.merge(stats, on="movie_id", how="inner").sort_values("movie_id").reset_index(drop=True)
    ids = catalogue.movie_id.to_numpy()
    positions = {int(movie): i for i, movie in enumerate(ids)}
    known = set(positions)
    content = create_similarity_matrix(create_genre_matrix(catalogue.rename(columns={"genres": "Genres"})))
    matrix = create_movie_user_matrix(create_user_movie_matrix(train)).reindex(ids)
    collaborative = create_collaborative_similarity(matrix)
    app_stats = catalogue.rename(columns={"movie_id": "MovieID", "title": "Title", "genres": "Genres"})
    popularity = recommend_popular_movies(app_stats, n=len(ids), min_ratings=100).MovieID.tolist()
    print("Training and similarities ready", flush=True)
    histories = {int(u): g for u, g in train.groupby("user_id")}
    positives = test[test.rating >= 4]
    rows = {name: [] for name in ["Popularity", "Genre similarity", "Collaborative filtering"]}
    recommended = {name: set() for name in rows}
    excluded = {"no_training_history": 0, "fewer_than_5_positive_training_ratings": 0, "no_eligible_unseen_test_positive": 0}
    eligible_positive_count = 0
    total_unseen_test_positive_count = 0
    for user, group in positives.groupby("user_id"):
        history = histories.get(int(user))
        if history is None:
            excluded["no_training_history"] += 1
            continue
        liked = history[history.rating >= 4].sort_values(["timestamp", "movie_id"])
        if len(liked) < 5:
            excluded["fewer_than_5_positive_training_ratings"] += 1
            continue
        seen = set(history.movie_id)
        unseen_positive = set(group.movie_id) - seen
        relevant = unseen_positive & known
        total_unseen_test_positive_count += len(unseen_positive)
        eligible_positive_count += len(relevant)
        if not relevant:
            excluded["no_eligible_unseen_test_positive"] += 1
            continue
        # Simulate a selected movie using only the user's last liked training movie.
        seed = int(liked.iloc[-1].movie_id)
        allowed = np.array([movie not in seen for movie in ids])
        recs = {"Popularity": [movie for movie in popularity if movie not in seen][:10]}
        for name, similarities in [("Genre similarity", content), ("Collaborative filtering", collaborative)]:
            scores = similarities[positions[seed]]
            order = np.argsort(-scores, kind="stable")
            recs[name] = [int(ids[i]) for i in order if allowed[i]][:10]
        for name, ranked in recs.items():
            assert not (set(ranked) & seen)
            assert set(ranked) <= known
            rows[name].append(ranking_metrics(ranked, relevant))
            recommended[name].update(ranked)
    users = len(rows["Popularity"])
    if not users:
        raise ValueError("No users meet evaluation eligibility.")
    metrics = []
    for name, values in rows.items():
        mean = np.mean(values, axis=0)
        metrics.append({"method": name, "users": users, "recall_at_10": float(mean[0]), "ndcg_at_10": float(mean[1]), "precision_at_10": float(mean[2]), "catalogue_coverage": len(recommended[name]) / len(ids), "unique_recommended_movies": len(recommended[name])})
    rng = np.random.default_rng(42)
    paired = np.array(rows["Collaborative filtering"])[:, 0] - np.array(rows["Popularity"])[:, 0]
    bootstrap = np.array([paired[rng.integers(0, users, users)].mean() for _ in range(2000)])
    report = {"protocol": {"train_fraction_target": 0.8, "cutoff_timestamp": cutoff, "cutoff_utc": pd.to_datetime(cutoff, unit="s", utc=True).isoformat(), "positive_rating_threshold": 4, "minimum_positive_training_ratings": 5, "query": "last positively rated training movie", "k": 10, "tie_break": "ascending movie ID", "popularity_min_ratings": 100, "test_tuning": False}, "data": {"ratings": len(ratings), "train_ratings": len(train), "test_ratings": len(test), "train_users": int(train.user_id.nunique()), "test_users": int(test.user_id.nunique()), "test_users_with_positive_ratings": int(positives.user_id.nunique()), "catalogue_movies": len(ids), "evaluated_users": users, "exclusions_among_test_positive_users": excluded, "eligible_unseen_test_positives": eligible_positive_count, "all_unseen_test_positives_for_history_qualified_users": total_unseen_test_positive_count}, "metrics": metrics, "paired_recall_difference_cf_minus_popularity": {"mean": float(paired.mean()), "bootstrap_95_percent_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(), "bootstrap_resamples": 2000, "seed": 42}, "source_sha256": {"ratings.dat": hashlib.sha256(ratings_path.read_bytes()).hexdigest(), "movies.dat": hashlib.sha256((raw_dir / "movies.dat").read_bytes()).hexdigest()}, "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(metrics).to_csv(output_dir / "evaluation_metrics.csv", index=False)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    evaluate(args.raw_dir, args.output_dir)
