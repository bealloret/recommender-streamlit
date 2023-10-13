import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, KNNBasic, accuracy
from surprise.model_selection import train_test_split

st.set_page_config(
    page_title='WBSFLIX-Recomender',
    page_icon='🎬',

)



st.title("WBSFLIX🎞️🍿🎬")
st.image('https://learn.wbscodingschool.com/wp-content/uploads/2022/07/CleanShot-2022-07-08-at-10.33.20@2x-1024x601.png')
st.write("Welcome to our Movie Recommender, the ultimate movie recommendation platform!Just for your entertainment, we have selected the top 10 movie recommendations based on our customer community's diverse preferences. We hope you enjoy these fantastic picks!")



# Load movies and ratings data
movies_df = pd.read_csv('movies.csv')  # Replace with the actual path to your movies.csv file
ratings_df = pd.read_csv('ratings.csv')  # Replace with the actual path to your ratings.csv file
rating_weight = 0.7
count_weight = 0.3
# Function for Popularity-Based Recommendations    
def popularity(ratings_df, movies_df):
    # Merge ratings_df and movies_df on 'movieId'
    df_1 = pd.merge(ratings_df, movies_df, on='movieId')

    # Group by 'movieId' and aggregate data
    test_df = df_1.groupby('movieId').agg({'rating': 'mean', 'userId': 'count'})

    # Scale the numeric columns (rating and userId)
    scaler = MinMaxScaler()
    test_df[['rating', 'userId']] = scaler.fit_transform(test_df[['rating', 'userId']])

    # Calculate weighted ratings
    test_df['book_rating_weighted'] = test_df['rating'] * rating_weight
    test_df['count_weighted'] = test_df['userId'] * count_weight
    test_df['final_rating'] = test_df['book_rating_weighted'] + test_df['count_weighted']

    # Sort by final_rating in descending order
    test_df = test_df.sort_values('final_rating', ascending=False)

    # Reset the index and merge with movie titles
    test_df = test_df.reset_index()
    output = test_df.merge(movies_df[['title', 'movieId']], how='left', on='movieId').drop_duplicates()

    return output

# Call the popularity function
popularity_recommendations = popularity(ratings_df, movies_df)

# Select specific columns from the DataFrame
selected_columns = popularity_recommendations[['movieId', 'title', 'final_rating']]

# Display the top 10 recommendations
top_10_recommendations = selected_columns.head(10)

# Streamlit UI
st.table(top_10_recommendations)




