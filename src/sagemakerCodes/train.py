# 라이브러리 가져오기
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import NearestNeighbors
import argparse
import joblib
import os


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_DATA", "/opt/ml/input/data/training"),
    )

    return parser.parse_known_args()


def train_and_save_model(args):
    print("------ directory ----")
    # 현재 작업 디렉토리 내의 파일 목록 얻기
    file_list = os.listdir("/opt/ml/input/data/training")
    print(file_list)

    # 결과 출력
    print("현재 작업 디렉토리의 파일 및 디렉토리 목록:")
    for file in file_list:
        print(file)

    #     game_info = np.load(os.path.join(args.data_dir, "games.npy"))
    #     titles = np.load(os.path.join(args.data_dir, "titles.npy"))
    game_info = np.load("/opt/ml/input/data/training/games.npy", allow_pickle=True)
    titles = np.load("/opt/ml/input/data/training/titles.npy", allow_pickle=True)
    reviews = np.load("/opt/ml/input/data/training/reviews.npy", allow_pickle=True)
    
    games_columns = np.load("/opt/ml/input/data/training/games_columns.npy", allow_pickle=True)
    reviews_columns = np.load("/opt/ml/input/data/training/reviews_columns.npy", allow_pickle=True)
    
    game_info = pd.DataFrame(game_info, columns=games_columns)
    reviews = pd.DataFrame(reviews, columns=reviews_columns)
    titles = pd.DataFrame(titles, columns=["app_id", "title"])
    
    merged_reviews = pd.merge(game_info, reviews, on="app_id")

    Y = merged_reviews["hours"]
    X = merged_reviews.drop(["hours"], axis=1)
    X = X[games_columns]

    model = DecisionTreeRegressor()
    model.fit(X, Y)

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print(titles)

    print("Model path: " + model_path)


if __name__ == "__main__":
    args, _ = _parse_args()
    print("--arguments--")
    print(args)

    train_and_save_model(args)
