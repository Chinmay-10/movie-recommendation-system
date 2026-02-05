"""
Data Preprocessing Module
Handles loading and preparing data for recommendation systems
"""

import pandas as pd
import numpy as np


def load_data(data_dir="../data"):
    """Load movies, ratings, and tags datasets"""
    movies = pd.read_csv(f"{data_dir}/movies.csv")
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    tags = pd.read_csv(f"{data_dir}/tags.csv")
    
    return movies, ratings, tags


def create_user_movie_matrix(ratings):
    """
    Create user-movie rating matrix
    
    Args:
        ratings: DataFrame with userId, movieId, rating columns
        
    Returns:
        user_movie_matrix: Pivot table with users as rows, movies as columns
    """
    user_movie_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )
    
    return user_movie_matrix


def get_movie_stats(ratings, movies):
    """
    Calculate movie statistics for recommendations
    
    Returns:
        DataFrame with movie stats including average rating and count
    """
    movie_stats = ratings.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    
    movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']
    movie_stats = movie_stats.merge(movies[['movieId', 'title', 'genres']], on='movieId')
    
    return movie_stats


def preprocess_genres(movies):
    """
    Preprocess genres for content-based filtering
    Converts pipe-separated genres to space-separated
    """
    movies_processed = movies.copy()
    movies_processed["genres"] = movies_processed["genres"].str.replace("|", " ", regex=False)
    
    return movies_processed