# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Read file
merged_df = pd.read_csv("datasets/merged_steam_games_.csv")

# Select from data
recommend_colums = ["app_id", "user_id", "is_recommended"]
recommend_df = merged_df[recommend_colums]
recommend_df.dropna(inplace=True)
print(recommend_df.head(10))

# Columns representing content characteristics
content_columns = merged_df.columns[31:58]
content_df = merged_df[content_columns]
content_df.dropna(inplace=True)
print(content_df.head(10))


scaler = StandardScaler()
content_df = pd.DataFrame(scaler.fit_transform(content_df), columns=content_columns)


def Recommend_Games(user_id):
    user_df = merged_df[recommend_df["user_id"] == user_id]

    if user_df.empty:
        return []

    # Vector representing the user's preference
    user_vector = user_df.iloc[:, 31:58].mean().values.reshape(1, -1)

    sim_scores = cosine_similarity(user_vector, content_df)

    rc_games = sim_scores.argsort()[0][::-1]
    unique_rc_games = set(merged_df["title"].iloc[rc_games])

    return list(unique_rc_games)[:10]


user_id = 7606333
recommendations = Recommend_Games(user_id)
print(f"10 Games Recommend to {user_id} : \n{recommendations}")


user_game_count = pd.DataFrame(merged_df.groupby("user_id")["app_id"].count())
# print(user_game_count[user_game_count['user_id'] > 10].shape)

over_10_users = user_game_count[user_game_count["app_id"] > 10].index
under_10_users = user_game_count[user_game_count["app_id"] <= 10].index
print(over_10_users)

train_X = []
test_X = []

for user in merged_df["user_id"].unique():
    X = merged_df[merged_df["user_id"] == user]

    if len(X) <= 10:
        train_X.append(X)
        continue

    train, test = train_test_split(X, test_size=0.2)
    train_X.append(train)
    test_X.append(test)

train_X = pd.concat(train_X)
test_X = pd.concat(test_X)

print(train_X.shape)
print(test_X.shape)

all_precisions = []
all_recalls = []
hitCount = 0

for user in over_10_users:
    user_test_X = test_X[test_X["user_id"] == user]
    actual_Y = pd.DataFrame(user_test_X[user_test_X["is_recommended"] == 1])

    predict_Y = pd.DataFrame(Recommend_Games(user), columns=["title"])

    if not actual_Y.empty and not predict_Y.empty:
        hit = len(set(actual_Y["title"]) & set(predict_Y["title"]))
        if hit > 0:
            hitCount += 1
        precision = hit / len(predict_Y["title"])
        recall = hit / len(actual_Y["title"])
        all_precisions.append(precision)
        all_recalls.append(recall)


average_precision = sum(all_precisions) / len(all_precisions)
average_recall = sum(all_recalls) / len(all_recalls)
HitRate = hitCount / len(over_10_users)

print("average_precision:", average_precision)
print("average_recall:", average_recall)
print("HitRate:", HitRate)
