import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import urllib.parse

def construct_query_url(base_url, path, params):
    query_string = urllib.parse.urlencode(params, doseq=True)
    return f"{base_url}{path}?{query_string}"

@st.cache_data()
def request_endpoint_game(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return None

def request_endpoint_recommend(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print("Response Data:", data)
        titles = data.get('titles')
        recommendations = data.get('recommendations')
        return titles, recommendations
    return None
page1_name = "게임 정보"
page2_name = "좋아하는 게임 선정 및 플레이 시간 예측"
st.sidebar.subheader("SageMaker Endpoint")
page = st.sidebar.selectbox("페이지", [page1_name, page2_name])
api_url = st.sidebar.text_input('fast api', value="3.38.216.206:8000")

st.title("Game Recommendations")

# 게임 정보
if st.sidebar.button('Get Game Information', key='get_game_info'):
    url = f"http://{api_url}/get_game_info"
    game_df = request_endpoint_game(url)
    if game_df is not None:
        st.write(game_df)
                
# 게임의 메인 카테고리
features = ['win', 'mac', 'linux', 'Action', 'Utilities', 'Animation & Modeling', 
            'Photo Editing', 'Education', 'Sports', 'Audio Production', 'Casual', 
            'Web Publishing', 'Accounting', 'Documentary', 'Sexual Content', 
            'Adventure', 'Video Production', 'Design & Illustration', 'Racing', 
            'Gore', 'Tutorial', 'RPG', 'Indie', 'Software Training', 'Simulation', 
            'Game Development', 'Massively Multiplayer', 'Early Access', 'Nudity', 
            'Strategy', 'Violent']
# 카테고리 선정
selected_features = st.multiselect("Select game features", features)
# 게임 추천 선정
if st.button("Get Recommendations", key='get_recommendations'):
    if len(selected_features) < 3:
        st.error("Please select at least three features")
    else:
        # 패스트 에이피아이 호출
        url = construct_query_url(f"http://{api_url}", "/recommend", {'features': selected_features})
        title, pred_df = request_endpoint_recommend(url)
        if title and pred_df is not None:
            st.subheader("Recommended Game")
            st.write(title[0])
            st.subheader("Recommended Play time about recommended game")
            st.write(pred_df)
        elif title:
            st.write("Recommendations:", title)
        else:
            st.write("No recommendations available.")
    