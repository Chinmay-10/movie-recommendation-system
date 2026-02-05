import pandas as pd
from content_model import ContentBasedRecommender
from collaborative_model import CollaborativeFilteringRecommender


class HybridRecommender:

    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings

        self.content = ContentBasedRecommender(movies)
        self.collaborative = CollaborativeFilteringRecommender(ratings, movies)

    def hybrid_recommend(self, user_id, reference_movie=None, top_n=10):

        results = []

        # ---------- CONTENT ----------
        if reference_movie is not None:
            content_df = self.content.get_recommendations(
                reference_movie, top_n
            )[["movieId"]].copy()

            content_df["content_score"] = (
                (top_n - content_df.index) / top_n
            )
            content_df["collab_score"] = 0.0
            content_df["source"] = "content"
            results.append(content_df)

        # ---------- COLLAB ----------
        collab_df = self.collaborative.get_recommendations(
            user_id, top_n
        )[["movieId"]].copy()

        collab_df["collab_score"] = (
            (top_n - collab_df.index) / top_n
        )
        collab_df["content_score"] = 0.0
        collab_df["source"] = "collaborative"
        results.append(collab_df)

        # ---------- COMBINE ----------
        hybrid = pd.concat(results, ignore_index=True)

        # ---------- FORCE METADATA ----------
        hybrid = hybrid.merge(
            self.movies[["movieId", "title", "genres"]],
            on="movieId",
            how="left"
        )

        # ---------- FINAL SCORE ----------
        hybrid["hybrid_score"] = (
            0.6 * hybrid["content_score"] +
            0.4 * hybrid["collab_score"]
        )

        return (
            hybrid
            .sort_values("hybrid_score", ascending=False)
            .drop_duplicates("movieId")
            .head(top_n)
            .reset_index(drop=True)
        )
