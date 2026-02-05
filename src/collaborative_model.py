"""
Collaborative Filtering Module
User-based collaborative filtering using cosine similarity
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringRecommender:
    """
    User-based collaborative filtering recommender
    """
    
    def __init__(self, ratings, movies):
        """
        Initialize the collaborative filtering recommender
        
        Args:
            ratings: DataFrame with userId, movieId, rating
            movies: DataFrame with movieId, title, genres
        """
        self.ratings = ratings
        self.movies = movies
        
        # Create user-movie matrix
        self.user_movie_matrix = ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating"
        )
        
        # Fill NaN with 0 for similarity calculation
        self.user_movie_matrix_filled = self.user_movie_matrix.fillna(0)
        
        # Compute user similarity
        self.user_similarity = cosine_similarity(self.user_movie_matrix_filled)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )
        
    def get_recommendations(self, user_id, num_recommendations=5):
        """
        Get collaborative filtering recommendations for a user
        
        Args:
            user_id: ID of the user
            num_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommended movies and predicted ratings
        """
        if user_id not in self.user_movie_matrix.index:
            print(f"User {user_id} not found")
            return pd.DataFrame()
        
        # Get similar users (top 10, excluding the user itself)
        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False)[1:11]
        
        # Get ratings from similar users
        similar_users_ratings = self.user_movie_matrix.loc[similar_users.index]
        
        # Calculate weighted average
        weighted_ratings = similar_users_ratings.T.dot(similar_users)
        normalization = similar_users.sum()
        
        predicted_ratings = weighted_ratings / normalization
        
        # Remove movies already rated by the user
        already_rated = self.user_movie_matrix.loc[user_id]
        predicted_ratings = predicted_ratings[already_rated.isna()]
        
        # Get top N recommendations
        top_recommendations = predicted_ratings.sort_values(ascending=False).head(num_recommendations)
        
        # Convert to DataFrame with movie details
        recommendations = pd.DataFrame({
            'movieId': top_recommendations.index,
            'predicted_rating': top_recommendations.values
        })
        
        recommendations = recommendations.merge(
            self.movies[['movieId', 'title', 'genres']], 
            on='movieId'
        )
        
        return recommendations
    
    def get_user_similarity(self, user_id1, user_id2):
        """Get similarity between two users"""
        try:
            return self.user_similarity_df.loc[user_id1, user_id2]
        except:
            return 0.0
    
    def get_user_rating_count(self, user_id):
        """Get number of ratings by a user"""
        if user_id in self.user_movie_matrix.index:
            return self.user_movie_matrix.loc[user_id].notna().sum()
        return 0