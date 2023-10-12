import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, KNNBasic
import os
from surprise.model_selection import train_test_split

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Define the file paths relative to the script directory
movies_file = os.path.join(script_dir, 'movies.csv')
ratings_file = os.path.join(script_dir, 'ratings.csv')

# Load data from CSV files
movies_df = pd.read_csv(movies_file)
ratings_df = pd.read_csv(ratings_file)

# Preprocess the data into the required format for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# Build a basic K-nearest neighbors (KNN) collaborative filtering model
sim_options = {
    'name': 'cosine',
    'user_based': False  # Item-based collaborative filtering
}
model = KNNBasic(sim_options=sim_options)

# Train the model on the entire dataset
trainset = data.build_full_trainset()
model.fit(trainset)

# Function for generating predictions for a user
def n_predictions_for_user(testset, userId, n):
    filtered_testset = []

    # Loop through the testset
    for row in testset:
        # Filter the ones with our user_id
        if row[0] == userId:
            # Add the filtered ones to an empty list
            filtered_testset.append(row)
    # Do predictions on the filtered testset
    predictions = model.test(filtered_testset)
   # Turn predictions into a dataframe
    prediction_df = pd.DataFrame(predictions)
    
    # Sort by 'est' in descending order and select the top 'n' predictions
    top_n_df = prediction_df.nlargest(n, 'est')
    
    # Filter the movie information
    filter_df = movies_df[['movieId', 'title', 'genres']]
    
    # Merge with the filter_df to include movie title and genres
    merged_df = top_n_df[['iid', 'est']].merge(filter_df, how='left', left_on='iid', right_on='movieId')
    
    # Drop duplicate movieId columns and reset the index
    merged_df = merged_df.drop(columns='movieId').reset_index(drop=True)
    
    return merged_df

# Create the Streamlit app
st.title("Movie Recommendation App")

# Input for User ID
user_id = st.selectbox("Select User ID", ratings_df['userId'].unique())

if user_id:
    num_recommendations = st.number_input("Number of Recommendations:", min_value=1, value=10)

    if st.button("Get Recommendations"):
        user_id = int(user_id)  # Convert to an integer
        testset = trainset.build_anti_testset()  # Create the testset
        user_predictions = n_predictions_for_user(testset, user_id, num_recommendations)
        st.dataframe(user_predictions)
    else:
        st.write("Please enter a valid User ID")


