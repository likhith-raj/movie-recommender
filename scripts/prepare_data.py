"""Build movie aggregates from locally obtained MovieLens 1M files."""
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def prepare(raw_dir, output):
    movies = pd.read_csv(raw_dir / "movies.dat", sep="::", engine="python",
                         names=["MovieID", "Title", "Genres"], encoding="latin-1")
    ratings = pd.read_csv(raw_dir / "ratings.dat", sep="::", engine="python",
                          names=["UserID", "MovieID", "rating", "timestamp"])
    stats = ratings.groupby("MovieID")["rating"].agg(avg_rating="mean", rating_count="count")
    result = movies.merge(stats, on="MovieID", how="inner")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Prepared {len(result)} rated movies from {len(ratings)} ratings: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/movie_stats.csv")
    args = parser.parse_args()
    prepare(args.raw_dir, args.output)
