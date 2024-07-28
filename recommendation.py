import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

# Correct path to CSV file
csv_file_path = 'C:/Users/Siddhi/OneDrive/Desktop/Siddhi pp/movie_recommendation_system/data/tmdb_5000_movies.csv'

# Load the movie dataset
movies_df = pd.read_csv(csv_file_path)

# Debug: Print the first few rows and columns of the dataframe
print("First few rows of the dataframe:")
print(movies_df.head())
print("\nColumns in the dataframe:")
print(movies_df.columns)

# Fill NaNs
movies_df['overview'] = movies_df['overview'].fillna('')
movies_df['genres'] = movies_df['genres'].fillna('')

# Function to preprocess genres
def preprocess_genres(genres):
    try:
        # Convert genres from JSON to a list of genre names
        genres_list = ast.literal_eval(genres)
        return ' '.join([genre['name'] for genre in genres_list])
    except:
        return ''

# Apply preprocessing to genres
movies_df['genres'] = movies_df['genres'].apply(preprocess_genres)

# Combine features into a single text field
movies_df['combined_features'] = movies_df['overview'] + ' ' + movies_df['genres']

# Debug: Print combined features to verify
print("\nCombined features sample:")
print(movies_df[['title', 'combined_features']].head())

# Create TF-IDF Vectorizer object and fit it to the movie combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['combined_features'])

# Debug: Print the shape of the TF-IDF matrix
print("\nTF-IDF matrix shape:")
print(tfidf_matrix.shape)

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(title, cosine_sim):
    # Get index of the movie with the given title
    idx = movies_df.index[movies_df['title'] == title].tolist()
    if not idx:
        print(f"Movie titled '{title}' not found.")
        return pd.DataFrame()  # Return an empty DataFrame if the title is not found
    idx = idx[0]
    
    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get the top 10 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    
    # Return a DataFrame of recommendations
    return movies_df[['title', 'vote_average']].iloc[movie_indices]

# Example: Generate recommendations for 'Backmask'
recommendations = get_recommendations('Avatar', cosine_sim)

# Debug: Check if recommendations are empty
if recommendations.empty:
    print('No recommendations found.')
else:
    print("\nRecommendations:")
    print(recommendations)

# Save recommendations to a JSON file
with open('recommendations.json', 'w') as f:
    json.dump(recommendations.to_dict(orient='records'), f, indent=4)
