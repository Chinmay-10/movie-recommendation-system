"""
Content-Based Recommendation Module
Recommends movies based on genre similarity using TF-IDF and cosine similarity
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Content-based recommender using TF-IDF on movie genres
    """
    
    def __init__(self, movies):
        """
        Initialize the content-based recommender
        
        Args:
            movies: DataFrame with movieId, title, and genres
        """
        self.movies = movies.copy()
        # Preprocess genres
        self.movies["genres"] = self.movies["genres"].str.replace("|", " ", regex=False)
        
        # Build TF-IDF matrix
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies["genres"])
        
        # Compute cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
    def get_recommendations(self, movie_title, num_recommendations=5):
        """
        Get content-based recommendations for a given movie
        
        Args:
            movie_title: Title of the movie
            num_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommended movies
        """
        try:
            # Get movie index
            idx = self.movies[self.movies["title"] == movie_title].index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top N similar movies (excluding the movie itself)
            sim_scores = sim_scores[1:num_recommendations+1]
            
            movie_indices = [i[0] for i in sim_scores]
            similarity_scores = [i[1] for i in sim_scores]
            
            # Return recommendations with scores
            recommendations = self.movies.iloc[movie_indices][['movieId', 'title', 'genres']].copy()
            recommendations['similarity_score'] = similarity_scores
            
            return recommendations
            
        except IndexError:
            print(f"Movie '{movie_title}' not found in database")
            return pd.DataFrame()
    
    def get_similarity_score(self, movie_title1, movie_title2):
        """Get similarity score between two movies"""
        try:
            idx1 = self.movies[self.movies["title"] == movie_title1].index[0]
            idx2 = self.movies[self.movies["title"] == movie_title2].index[0]
            return self.cosine_sim[idx1][idx2]
        except:
            return 0.0