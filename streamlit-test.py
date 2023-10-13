import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, KNNBasic
import os
from surprise.model_selection import train_test_split
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Define the file paths relative to the script directory
movies_file = os.path.join(script_dir, 'movies.csv')
ratings_file = os.path.join(script_dir, 'ratings.csv')

# Load data from CSV files
movies_df = pd.read_csv(movies_file, decimal = '.')
ratings_df = pd.read_csv(ratings_file, decimal = '.')

# Calculate the cosine similarity matrix for movies
user_movie_matrix = pd.pivot_table(ratings_df, index='userId', columns='movieId', values='rating')
user_movie_matrix = user_movie_matrix.fillna(user_movie_matrix.mean())
movies_cosines_matrix = pd.DataFrame(cosine_similarity(user_movie_matrix.T),
                                    columns=user_movie_matrix.columns,
                                    index=user_movie_matrix.columns)



# Create a dropdown for selecting the recommender
recommender = st.selectbox("Select Recommender", ["Recommender 1", "Recommender 2", "Recommender 3"])

# Depending on the selected recommender, perform specific actions
if recommender == "Recommender 1":
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

        # Rename columns to match your specifications
        merged_df = merged_df.rename(columns={'iid': 'movieId', 'est': 'rating (estimated)'})

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

if recommender == "Recommender 2":
    def popularity_rat1(number_rating_m, rating_m):
        #Creating the new df for our calculation
        new_df = ratings_df.groupby('movieId').agg({'rating':['mean', 'count']})
    
        # We change the name of the new columns
        new_df.columns = ['rating_mean', 'rating_count']
        scaler = MinMaxScaler()
        new_df[['rating_mean', 'rating_count']] = scaler.fit_transform(new_df[['rating_mean', 'rating_count']])
    
        #We create different rates for the variables that are going to be the base of the recommendation 
        weight_rating_count = 0.3
        weight_average_rating = 0.7
    
        #We create the function to calculate the final rating
        new_df['popularity_score'] = (
        weight_rating_count * new_df['rating_mean'] +
        weight_average_rating * new_df['rating_count']
        )
        
        #Reorganising the table
        new_df = new_df.sort_values('popularity_score',ascending=False)
        new_df = new_df.reset_index()
        new_df = new_df.merge(movies_df[['title','genres','movieId']],how='left',on='movieId').drop_duplicates()
        
        return new_df[['movieId','title','genres','popularity_score']].head(5)

    result = popularity_rat1('rating_count', 'rating_mean')
    st.dataframe(result)

if recommender == "Recommender 3":
    # Input for Movie ID
    movie_id = st.number_input("Enter a Movie ID (1-9742):", min_value=1, max_value=9742)
    n = 15  # Define 'n' here or obtain it from user input
    
    if st.button("Get Recommendations"):
        if movie_id:
            # Define the item_similarity function here
            def item_similarity(movieId, n):
                # Create a DataFrame using the values from 'movies_cosines_matrix' for the 'movieId'.
                cosines_df = pd.DataFrame(movies_cosines_matrix[movieId])

                # Remove the row with the index 
                cosines_df = cosines_df[cosines_df.index != movieId]

                # Sort the 'lovely_bones_cosines_df' by the column 'lovely_bones_cosine' column in descending order.
                cosines_df = cosines_df.sort_values(by=movieId, ascending=False)

                # Find out the number of users rated two movies
                no_of_users_rated_two_movies = [sum((user_movie_matrix[movieId] > 0) & (user_movie_matrix[x] > 0)) for x in cosines_df.index]
                # Create a column for the number of users who rated the movie and another movie
                cosines_df['users_who_rated_two_movies'] = no_of_users_rated_two_movies

                # Remove recommendations that have less than 10 users who rated two movies
                cosines_df = cosines_df[cosines_df["users_who_rated_two_movies"] > 10]

                # Get the title and genre for the related movies
                movie_info_columns = ['movieId', 'title', 'genres']

                top_n_cosine = (cosines_df
                                  .head(n)   # this would give us the top n movie-recommendations
                                  .reset_index()
                                  .merge(movies_df.drop_duplicates(subset='movieId'),
                                          on='movieId',
                                          how='left')
                                  [movie_info_columns + [movieId, 'users_who_rated_two_movies']]
                                  )

                return top_n_cosine

            # Call the item_similarity function with the specified movie_id and n
            result3 = item_similarity(movie_id, n)
            st.dataframe(result3)
        else:
            st.write("Please enter a valid Movie ID.")
