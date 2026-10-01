# MovieLens Movie Recommender

A Streamlit movie-discovery app with genre-based similar-movie recommendations and a weighted-rating popularity baseline. A separate Python module implements item-to-item collaborative filtering.

## What is implemented

- **Content-based retrieval:** multi-hot genre encoding and cosine similarity for a selected movie.
- **Popularity baseline:** weighted average ratings with a minimum rating-count threshold of 100 in the app.
- **Collaborative filtering module:** item-item cosine similarity over user-rating vectors, with missing ratings filled with zero. This module is not yet connected to the app.
- **Streamlit interface:** select a movie, inspect similar titles and browse popular movies.

The app does not use a user's rating history. Movie selection is a query, not personalised user-profile modelling. No hybrid model is implemented. A chronological held-out baseline evaluation is documented below.

## Setup

Use Python 3.12 and run these commands from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

Download [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) from GroupLens and extract `movies.dat` and `ratings.dat` into `data/raw/`. Then run:

```powershell
python scripts/prepare_data.py
python -m streamlit run app/app.py
```

The preparation script builds `data/processed/movie_stats.csv` from the raw files. It keeps rated movies and calculates average ratings and rating counts. Raw and processed data are excluded from Git.

## Methods

For popularity, each eligible movie is ranked by:

`weighted_rating = (v / (v + m)) * R + (m / (v + m)) * C`

Here `v` is its rating count, `R` its mean rating, `m` the minimum rating count, and `C` the unweighted mean of movie-average ratings. This is a shrinkage-style baseline, not a learned ranking model.

Genre similarity is cosine similarity between binary genre vectors. Collaborative similarity uses movie rating vectors across users. Both exclude the selected movie explicitly, including when scores tie.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests check self-exclusion under similarity ties, non-default table indexes, ambiguous titles, unknown titles, zero-result requests and the popularity threshold. These are correctness tests, not a measurement of recommendation quality.

## Repository layout

```text
app/app.py                       Streamlit interface
src/content_recommender.py       Content and popularity methods
src/collaborative_recommender.py Item-item collaborative method
scripts/prepare_data.py          Reproducible data preparation
tests/test_recommendations.py   Regression tests
data/raw/                        Locally obtained MovieLens files
data/processed/                  Generated aggregates
```

## Limitations and evaluation plan

- Genre vectors are coarse; many movies tie and their order follows input order.
- Full similarity matrices use quadratic memory in the number of movies.
- Zero-filled raw ratings do not account for individual users' rating scales or minimum co-rating support.
- Popularity uses historical aggregate ratings, not recency or current trends.
- Evaluation is limited to warm-start users and a simulated selected-movie query; no online engagement testing has been conducted.

## Offline evaluation

A global chronological split of MovieLens 1M evaluated all 1,079 eligible warm-start users using their last liked training movie as a simulated query. The Streamlit app itself does not use user histories.

| Method | Recall@10 | NDCG@10 | Catalogue coverage |
|---|---:|---:|---:|
| Weighted popularity | 0.0376 | 0.1633 | 2.48% |
| Genre similarity | 0.0042 | 0.0215 | 31.57% |
| Item-item collaborative | 0.0354 | 0.1203 | 50.08% |

Popularity had the strongest observed average ranking scores; collaborative filtering offered broader coverage. The paired recall difference interval includes zero. These scores are not accuracy percentages or evidence of online/business improvement.

Run `python scripts/evaluate.py` after obtaining raw data. See [full protocol, limitations and results](reports/EVALUATION.md) and [machine-readable results](reports/evaluation_results.json).

Next work: design a validation split before tuning tie-breaking or hybrid weights, and use a new untouched test period for final comparison. Collaborative UI integration and a hybrid remain future work.

## Dataset acknowledgement

This project uses MovieLens 1M from GroupLens Research. See the [dataset terms](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt). Do not redistribute its data without separate permission. Commercial or revenue-bearing use also requires permission. No GroupLens endorsement is implied.

Dataset citation: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems 5(4), Article 19. https://doi.org/10.1145/2827872
