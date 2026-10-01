# Offline recommendation evaluation

## Result

The weighted-rating popularity baseline had the highest observed mean Recall@10 and NDCG@10. Item-item collaborative filtering covered substantially more of the catalogue. The paired user bootstrap interval for collaborative-minus-popularity recall includes zero, so this run does not establish a reliable recall improvement for either method over the other.

| Method | Recall@10 | NDCG@10 | Precision@10 | Catalogue coverage |
|---|---:|---:|---:|---:|
| Popularity | 0.0376 | 0.1633 | 0.1487 | 2.48% |
| Genre similarity | 0.0042 | 0.0215 | 0.0209 | 31.57% |
| Collaborative filtering | 0.0354 | 0.1203 | 0.1131 | 50.08% |

## Protocol

- Data: MovieLens 1M, 1,000,209 ratings.
- Global chronological split at 2000-12-02 14:52:18 UTC: 800,164 training ratings and 200,045 test ratings. Equal-timestamp events remain on the same side.
- Population: all 1,079 eligible users; no user sampling. Require at least five training ratings of 4 or 5 stars and at least one unseen, positively rated test movie in the training catalogue.
- Among 1,762 test users with positive ratings, exclude 640 without training history and 43 with fewer than five positive training ratings. This is a warm-start evaluation.
- Query: each user's most recently positively rated training movie. This simulates selecting a known liked title; it does not add user-history personalisation to the Streamlit app.
- Ranking: return up to ten unseen movies. Content/CF candidates cover all 3,662 movies with training ratings. Popularity retains the existing minimum of 100 training ratings. Identical user histories and relevant targets are used for all methods.
- Relevance: held-out ratings of 4 or 5 stars. Exclude all previously rated titles and 42 cold-start positive items from the relevance denominator. There are 53,764 eligible positive user-movie pairs.
- Fit popularity statistics and collaborative similarities on training ratings only. Genre metadata is static. Do not load the full-data processed CSV for evaluation.
- Genre and collaborative ties resolve by ascending MovieID. All method choices were fixed before viewing results; no test-set tuning.

## Metric definitions

- Recall@10: relevant held-out titles retrieved divided by all eligible relevant titles for that user, averaged equally across users.
- NDCG@10: binary relevance discounted by log2(rank + 1), divided by the ideal top-ten ranking for that user; average equally across users.
- Precision@10: number of relevant retrieved titles divided by ten, averaged across users.
- Catalogue coverage: distinct recommended titles across evaluated users divided by 3,662. This measures breadth, not relevance or accuracy.

## Uncertainty and interpretation

Collaborative-minus-popularity mean Recall@10 difference: -0.002185. Paired bootstrap 95% interval: [-0.008361, 0.004188], using 2,000 resamples with seed 42.

The interval is conditional on this user population and fixed temporal split; it does not capture uncertainty across time periods. No statistical interval was calculated for NDCG or coverage. Popularity has stronger observed NDCG, while CF spreads recommendations across more titles. Genre-only vectors create many ties; the deterministic ID ordering can strongly affect that baseline. These results do not show a production or business impact.

Held-out ratings are incomplete evidence of preferences: an unrated recommendation is treated as a non-hit, not a proven dislike. Users may have many positive held-out movies, which reduces Recall@10 mechanically. This setup evaluates future liked-item retrieval from a simulated seed, not a human judgement of movie-to-movie similarity. The ratings reflect historical MovieLens behaviour, not present-day engagement.

## Reproduce

Obtain MovieLens 1M directly from GroupLens, place movies.dat and ratings.dat in data/raw, install requirements, then run from the repository root:

```powershell
python -m unittest discover -s tests -v
python scripts/evaluate.py
```

Machine-readable results, package versions and source-file SHA256 hashes are saved in evaluation_results.json. Summary scores are in evaluation_metrics.csv. No raw ratings or user-level outputs are published.

## Next experiment

Design a validation split before tuning genre tie-breaking, combining multiple liked seeds or hybrid weights. Retain an untouched later test period for final comparison. A new experiment should not repeatedly optimise against this already-inspected test set.

Metric reference: [scikit-learn NDCG documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html). Temporal leakage reference: [A Critical Study on Data Leakage in Recommender System Offline Evaluation](https://arxiv.org/abs/2010.11060).
