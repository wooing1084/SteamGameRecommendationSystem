#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt

#Read file
merged_df = pd.read_csv("datasets/merged_steam_games_.csv")

#Select from data
recommend_colums = ['app_id', 'user_id', 'is_recommended']
recommend_df = merged_df[recommend_colums]
recommend_df.dropna(inplace = True)
print(recommend_df.head(10))

#Columns representing content characteristics
content_columns = merged_df.columns[31:58]
content_df = merged_df[content_columns]
content_df.dropna(inplace = True)
print(content_df.head(10))

#Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
content_df = pd.DataFrame(scaler.fit_transform(content_df), columns=content_columns)

def Recommend_Games(user_id):
    user_df = merged_df[recommend_df['user_id'] == user_id]
    
    if user_df.empty:
        return []

    #Vector representing the user's preference
    user_vector = user_df.iloc[:,31:58].mean().values.reshape(1, -1)
    
    from sklearn.metrics.pairwise import cosine_similarity
    sim_scores = cosine_similarity(user_vector, content_df)
    
    rc_games = sim_scores.argsort()[0][::-1]
    unique_rc_games = set(merged_df['title'].iloc[rc_games])
    
    return list(unique_rc_games)[:10]

user_id = 7606333
recommendations = Recommend_Games(user_id)
print(f"10 Games Recommend to {user_id} : \n{recommendations}")