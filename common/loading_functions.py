import os
from typing import Any, Tuple

import pandas as pd

from config import CONFIG


def loading_movielens_1m(file_path):
    ratings_header = "UserID::MovieID::Rating::Timestamp"
    movies_header = "MovieID::Title::Genres"
    user_header = "UserID::Gender::Age::Occupation::Zip-code"

    ratings = pd.read_csv(
        os.path.join(file_path, 'ratings.dat'),
        sep='::', header=None, names=ratings_header.split('::'),
        engine='python'
    )

    movies = pd.read_csv(
        os.path.join(file_path, 'movies.dat'),
        sep='::', header=None, names=movies_header.split('::'),
        engine='python', encoding='iso-8859-1'
    )

    users = pd.read_csv(
        os.path.join(file_path, 'users.dat'),
        sep='::', header=None, names=user_header.split('::'),
        engine='python', encoding='iso-8859-1'
    )

    ratings.sort_values(['UserID', 'Timestamp'], inplace=True)

    # MovieID -> item_id
    org_movie_id = set(ratings['MovieID'].unique().tolist() + movies['MovieID'].unique().tolist())
    movie_id_mapper = {
        movie_id: item_id for item_id, movie_id in enumerate(org_movie_id)
    }

    ratings['item_id'] = ratings['MovieID'].map(lambda x: movie_id_mapper[x])
    movies['item_id'] = movies['MovieID'].map(lambda x: movie_id_mapper[x])

    # UserID -> user_id
    org_user_id = set(ratings['UserID'].unique().tolist() + users['UserID'].unique().tolist())
    user_id_mapper = {
        user_id: user_index_id for user_index_id, user_id in enumerate(org_user_id)
    }

    ratings['user_id'] = ratings['UserID'].map(lambda x: user_id_mapper[x])
    users['user_id'] = users['UserID'].map(lambda x: user_id_mapper[x])

    return ratings, movies, users


def loading_data(data_type: str) -> Tuple[Any, Any, Any]:
    if data_type == '1M':
        file_path = os.path.join(CONFIG.DATA, 'movielens', 'ml-1m')
        loading_function = loading_movielens_1m
    else:
        raise ValueError(f"unknown data type {data_type}")

    return loading_function(file_path)
