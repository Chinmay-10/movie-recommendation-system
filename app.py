import streamlit as st
import sys

sys.path.append("src")

from preprocess import load_data
from hybrid import HybridRecommender
from explainability import visualize_recommendation_scores

st.set_page_config(page_title="Movie Recommender", layout="wide")


@st.cache_resource
def load_system():
    movies, ratings, _ = load_data("data")
    recommender = HybridRecommender(movies, ratings)
    return recommender, movies, ratings


def main():
    st.title("🎬 Movie Recommendation System")

    recommender, movies, ratings = load_system()

    st.sidebar.header("Configuration")

    user_id = st.sidebar.selectbox(
        "Select User ID",
        sorted(ratings["userId"].unique())
    )

    reference_movie = st.sidebar.selectbox(
        "Reference Movie (Optional)",
        ["None"] + sorted(movies["title"].unique())
    )

    if reference_movie == "None":
        reference_movie = None

    num_recs = st.sidebar.slider(
        "Number of Recommendations",
        5, 20, 10
    )

    if st.button("Generate Recommendations"):
        recs = recommender.hybrid_recommend(  # ← CHANGED THIS
            user_id=user_id,
            reference_movie=reference_movie,
            top_n=num_recs  # ← CHANGED THIS
        )

        if recs.empty:
            st.warning("No recommendations found.")
            return

        st.subheader("Recommended Movies")
        
        for i, row in recs.iterrows():
            st.write(f"**{i + 1}. {row['title']}**")
            st.caption(f"Genres: {row['genres']}")

        st.subheader("Score Breakdown")
        fig = visualize_recommendation_scores(recs)
        st.pyplot(fig)


if __name__ == "__main__":
    main()