import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import urllib.parse

# Function to construct a query URL with a base URL, path, and parameters.
def construct_query_url(base_url, path, params):
    # Encode parameters into a query string
    query_string = urllib.parse.urlencode(params, doseq=True)
    return f"{base_url}{path}?{query_string}"

# Function to request game information from an endpoint and return it as a DataFrame.
@st.cache_data()
def request_endpoint_game(url):
    # Send a GET request to the specified URL
    response = requests.get(url)
    if response.status_code == 200:
        # If the response status code is 200, return the JSON data as a DataFrame
        return pd.DataFrame(response.json())
    return None

# Function to request game recommendations from an endpoint and return titles and recommendations.
def request_endpoint_recommend(url):
    # Send a GET request to the specified URL
    response = requests.get(url)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        print("Response Data:", data)
        # Extract titles and recommendations from the response
        titles = data.get('titles')
        recommendations = data.get('recommendations')
        return titles, recommendations
    return None

page1_name = "Game Information"
page2_name = "Pick your favorite games and predict play time"
st.sidebar.subheader("SageMaker Endpoint")
page = st.sidebar.selectbox("Page", [page1_name, page2_name])
api_url = st.sidebar.text_input('fast api', value="15.165.43.140:8000")

if page == page1_name:
    # Display title for Game Information page
    st.title("Game Information")

    # Retrieve and display game information
    if st.sidebar.button('Get Game Information', key='get_game_info'):
        # Construct the URL for fetching game information
        url = f"http://{api_url}/get_game_info"
        # Request game information from the endpoint
        game_df = request_endpoint_game(url)
        if game_df is not None:
            # Display the game information as a DataFrame
            st.write(game_df)

elif page == page2_name:
    # Display title for the page to select favorite games and predict playtime
    st.title("Select Your Favorite Games and Predict Play Time")

    # List of game categories
    features = ['win', 'mac', 'linux', 'Action', 'Utilities', 'Animation & Modeling', 
                'Photo Editing', 'Education', 'Sports', 'Audio Production', 'Casual', 
                'Web Publishing', 'Accounting', 'Documentary', 'Sexual Content', 
                'Adventure', 'Video Production', 'Design & Illustration', 'Racing', 
                'Gore', 'Tutorial', 'RPG', 'Indie', 'Software Training', 'Simulation', 
                'Game Development', 'Massively Multiplayer', 'Early Access', 'Nudity', 
                'Strategy', 'Violent']

    # Select game categories
    selected_features = st.multiselect("Select game features", features)

    # Get game recommendations
    if st.sidebar.button("Get Recommendations", key='get_recommendations'):
        if len(selected_features) >= 3:
            # Call the FastAPI endpoint to get game recommendations
            url = construct_query_url(f"http://{api_url}", "/recommend", {'features': selected_features})
            title, pred_df = request_endpoint_recommend(url)
            print(title, pred_df)
            if title and pred_df is not None:
                # Display the recommended game title and predicted playtime
                st.subheader("Recommended Game")
                st.write(title[0])
                st.subheader("Recommended Playtime for the Recommended Game")
                st.write(pred_df)
            elif title:
                # Display recommendations if available
                st.write("Recommendations:", title)
            else:
                # Display a message when no recommendations are available
                st.write("No recommendations available.")

    