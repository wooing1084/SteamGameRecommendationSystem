# SteamGameRecommendationSystem

가천대학교 2023학년 2학기 머신러닝 


We have created a game recommendation system for Steam using various machine learning models.  
Also implemented a simple service using AWS SageMaker, Cloud9, and Streamlit.

## Data Reference
 [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam?select=users.csv)

## How to run?
### Recommendation models
- Download datasets from the reference
- Run "preprocessing.ipynb"
- Just run these jupyter notebooks
  - content_based.ipynb
  - item_based.ipynb
  - SVD.ipynb
  - KNN.ipynb
  - LogisticRegression.ipynb
  
### Stremlit
- Make virtual environment
- Install libraries  
  - pip install streamlit==1.28.2 urllib3==1.26.6 matplotlib==3.7.3 pandas==2.0.3
- Run the main.py
(But The server has shut downe)
