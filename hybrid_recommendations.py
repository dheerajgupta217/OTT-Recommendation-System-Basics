import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("ott_ratings.csv")

# Create user-item matrix
user_item_matrix = df.pivot_table(index='user_id', columns='show_id', values='rating')

# Fill missing values with 0
user_item_matrix_filled = user_item_matrix.fillna(0)

# Compute similarity between users
user_similarity = cosine_similarity(user_item_matrix_filled)

# Convert to DataFrame
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

print("User Similarity Matrix:\n", user_similarity_df)

# Function to recommend shows
def recommend_shows(user_id, num_recommendations=2):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    
    recommended_shows = set()
    
    for similar_user in similar_users.index:
        shows = user_item_matrix.loc[similar_user].dropna().index
        recommended_shows.update(shows)
        
        if len(recommended_shows) >= num_recommendations:
            break

    # Remove already watched shows
    watched = user_item_matrix.loc[user_id].dropna().index
    recommendations = list(recommended_shows - set(watched))
    
    return recommendations[:num_recommendations]

# Example
user_id = 1
recommendations = recommend_shows(user_id)

print(f"Recommended shows for user {user_id}:", recommendations)
