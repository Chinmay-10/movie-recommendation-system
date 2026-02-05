"""
Explainability Module
Provides human-readable explanations and visualizations for recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def format_explanation(explanation_dict):
    """
    Format explanation dictionary into readable text
    
    Args:
        explanation_dict: Dictionary from HybridRecommender.explain_recommendation()
        
    Returns:
        Formatted explanation string
    """
    movie = explanation_dict['movie']
    reasons = explanation_dict['reasons']
    
    output = f"\n🎬 Why we recommended '{movie}':\n"
    output += "=" * 60 + "\n"
    
    for i, reason in enumerate(reasons, 1):
        output += f"{i}. {reason}\n"
    
    return output


def explain_hybrid_weights(user_rating_count, content_weight, collab_weight):
    """
    Explain the weighting strategy used in hybrid recommendation
    
    Args:
        user_rating_count: Number of ratings by the user
        content_weight: Weight given to content-based filtering
        collab_weight: Weight given to collaborative filtering
        
    Returns:
        Explanation string
    """
    explanation = "\n📊 Recommendation Strategy:\n"
    explanation += "=" * 60 + "\n"
    
    if user_rating_count < 5:
        explanation += "🆕 NEW USER DETECTED (Cold-Start Problem)\n"
        explanation += f"   • You have only {user_rating_count} rating(s)\n"
        explanation += f"   • Content-based weight: {content_weight*100:.1f}%\n"
        explanation += f"   • Collaborative weight: {collab_weight*100:.1f}%\n"
        explanation += "   • Strategy: Emphasizing genre similarity since we don't have\n"
        explanation += "     enough data about your preferences yet.\n"
    elif user_rating_count < 20:
        explanation += "👤 GROWING USER PROFILE\n"
        explanation += f"   • You have {user_rating_count} ratings\n"
        explanation += f"   • Content-based weight: {content_weight*100:.1f}%\n"
        explanation += f"   • Collaborative weight: {collab_weight*100:.1f}%\n"
        explanation += "   • Strategy: Balanced approach using both genre similarity\n"
        explanation += "     and user behavior patterns.\n"
    else:
        explanation += "⭐ ESTABLISHED USER PROFILE\n"
        explanation += f"   • You have {user_rating_count} ratings\n"
        explanation += f"   • Content-based weight: {content_weight*100:.1f}%\n"
        explanation += f"   • Collaborative weight: {collab_weight*100:.1f}%\n"
        explanation += "   • Strategy: Leveraging collaborative filtering based on\n"
        explanation += "     users with similar taste to yours.\n"
    
    return explanation


def visualize_recommendation_scores(recommendations_df, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    # ✅ SAFETY: handle missing score columns
    if "content_score" not in recommendations_df.columns:
        recommendations_df["content_score"] = 0.0

    if "collab_score" not in recommendations_df.columns:
        recommendations_df["collab_score"] = 0.0

    movies = recommendations_df["title"].str[:30]
    content_scores = recommendations_df["content_score"]
    collab_scores = recommendations_df["collab_score"]

    x = np.arange(len(movies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width/2, content_scores, width, label="Content-Based")
    ax.bar(x + width/2, collab_scores, width, label="Collaborative")

    ax.set_xticks(x)
    ax.set_xticklabels(movies, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Recommendation Score Breakdown")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig