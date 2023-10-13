import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, KNNBasic, accuracy
from surprise.model_selection import train_test_split



st.title("WBSFLIX🎞️🍿🎬")
st.image('https://telegraphstar.com/wp-content/uploads/2020/04/9x-1920x1182.jpg')
st.subheader('Welcome to our Movie Recommender')




###### all my functions

def item_similarity(movieId, n):
    # Create a DataFrame using the values from 'movies_cosines_matrix' for the given movieId
    cosines_df = pd.DataFrame(movies_cosines_matrix[movieId])

    # Remove the row with the index matching the input movieId
    cosines_df = cosines_df[cosines_df.index != movieId]

    # Sort the cosine similarity values in descending order
    cosines_df = cosines_df.sort_values(by=movieId, ascending=False)

    # Find out the number of users who rated both the input movie and the other movies
    no_of_users_rated_both_movies = [sum((user_movie_matrix[movieId] > 0) & (user_movie_matrix[x] > 0)) for x in cosines_df.index]

    # Create a column for the number of users who rated both movies
    cosines_df['users_who_rated_both_movies'] = no_of_users_rated_both_movies

    # Remove recommendations that have less than 10 users who rated both movies
    cosines_df = cosines_df[cosines_df['users_who_rated_both_movies'] > 10]

    # Get the names and other information for the recommended movies
    top_n_cosine = (cosines_df
                            .head(n)
                            .reset_index()
                            .merge(df_1.drop_duplicates(subset='movieId'),
                                    on='movieId',
                                    how='left')
                            [movie_info_columns + [movieId, 'users_who_rated_both_movies']]
                            )

    return top_n_cosine


def n_predictions_for_user(testset, user_input, n):
    filtered_testset = []

    # Loop through a big testset
    for row in testset:
        # Filter the ones with our user_id
        if row[0] == user_input:
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

##### end of all functions
# Load movies and ratings data
movies_df = pd.read_csv('movies.csv')  # Replace with the actual path to your movies.csv file
ratings_df = pd.read_csv('ratings.csv')  # Replace with the actual path to your ratings.csv file

# Calculate the cosine similarity matrix for movies
user_movie_matrix = pd.pivot_table(ratings_df, index='userId', columns='movieId', values='rating')
user_movie_matrix = user_movie_matrix.fillna(value = 0)
movies_cosines_matrix = pd.DataFrame(cosine_similarity(user_movie_matrix.T),
                                    columns=user_movie_matrix.columns,
                                    index=user_movie_matrix.columns)


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


st.header("Choose Your Recommender")
model = st.selectbox("Pick Your Movie Magic Generator", ["Similar Movies", "Similar People Like You"])

if model == "Similar Movies":
    user_input = st.text_input("Enter a Movie ID:")

    num_recommendations = st.number_input("Number of Recommendations:", min_value=1, value=10)

    if st.button("Get Recommendations"):
        if user_input:
            df_1 = pd.merge(ratings_df, movies_df, on='movieId')
                    
            user_movie_matrix = pd.pivot_table(data=df_1,
                                  values='rating',
                                  index='userId',
                                  columns='movieId',
                                  fill_value=0)
                    
            movies_cosines_matrix = pd.DataFrame(cosine_similarity(user_movie_matrix.T),
                                        columns=user_movie_matrix.columns,
                                        index=user_movie_matrix.columns)
            movie_info_columns = ['movieId', 'title']
                    
            # Call the item_similarity function with user_input and num_recommendations
            recommendations = item_similarity(int(user_input), num_recommendations)
                    
            st.header("Recommended Movies:")
            st.table(recommendations)

elif model == "Similar People Like You":
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

    user_id = st.selectbox("Select a User ID", ratings_df['userId'].unique())

    if user_id:
        num_recommendations = st.number_input("Number of Recommendations:", min_value=1, value=10)

        if st.button("Get Recommendations"):
            user_id = int(user_id)  # Convert to an integer
            testset = trainset.build_anti_testset()  # Create the testset
            user_predictions = n_predictions_for_user(testset, user_id, num_recommendations)
            st.dataframe(user_predictions)
        else:
            st.write("Please enter a valid User ID")










        