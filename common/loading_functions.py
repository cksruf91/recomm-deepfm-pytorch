import json
import os
import random
import re
from itertools import accumulate
from typing import Any, Tuple

import pandas as pd
from pandas import DataFrame

from common.utils import progressbar, to_timestampe
from config import CONFIG

random.seed(42)


def train_test_split(df: DataFrame, user_col: str, time_col: str) -> Tuple[DataFrame, DataFrame]:
    """ 학습 테스트 데이터 분리 함수
    각 유저별 마지막 interaction 읕 테스트로 나머지를 학습 데이터셋으로 사용

    Args:
        df: 전체 데이터
        user_col: 기준 유저 아이디 컬럼명
        time_col: 기준 아이템 아이디 컬럼명

    Returns: 학습 데이터셋, 테스트 데이터셋
    """
    
    # temp = df.reset_index()
    # test_idx = temp.groupby(user_col)['index'].last().tolist()
    # train_idx = list(set(df.index) - set(test_idx))
    # test = df.loc[test_idx]
    # train = df.loc[train_idx]

    last_action_time = df.groupby(user_col)[time_col].transform('max')

    test = df[df[time_col] == last_action_time]
    train = df[df[time_col] != last_action_time]

    test = test.groupby(user_col).first().reset_index()

    print(f'test set size : {len(test)}')
    user_list = train[user_col].unique()
    drop_index = test[test[user_col].isin(user_list) == False].index
    test.drop(drop_index, inplace=True)
    print(f'-> test set size : {len(test)}')

    return train, test


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
    
    ratings.sort_values(['UserID','Timestamp'], inplace=True)

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


def uniform_random_sample(n, exclude_items, items):
    sample = []
    while len(sample) < n:
        n_item = random.choice(items)
        if n_item in exclude_items:
            continue
        if n_item in sample:
            continue
        sample.append(n_item)
    assert len(sample) == n
    return sample


def weighted_random_sample(n, exclude_items, items, cum_sums):
    n_items = len(exclude_items)
    samples = random.choices(
        items, cum_weights=cum_sums, k=n_items + n + 100
    )

    sample = list(set(samples) - exclude_items)
    sample = sample[:n]
    assert len(sample) == n
    return sample


def get_negative_samples(train_df, test_df, user_col, item_col, n_sample=99, method='random'):
    negative_sampled_test = []

    # 샘플링을 위한 아이템들의 누적합
    # train_df.loc[:,'item_count'] = 1
    train_df = train_df.assign(item_count=1)
    item_counts = train_df.groupby(item_col)['item_count'].sum().reset_index()
    item_counts['cumulate_count'] = [c for c in accumulate(item_counts.item_count)]

    # 샘플링을 위한 변수
    item_list = item_counts[item_col].tolist()
    item_cumulate_count = item_counts['cumulate_count'].tolist()

    # 유저가 이전에 interaction 했던 아이템들
    user_interactions = train_df.groupby(user_col)[item_col].agg(lambda x: set(x.tolist()))

    for uid, iid in zip(test_df[user_col].tolist(), test_df[item_col].tolist()):
        row = [uid, iid]

        try:
            inter_items = user_interactions[uid]
        except KeyError as e:
            inter_items = set([])

        if method == 'random':
            sample = uniform_random_sample(n_sample, inter_items, item_list)
        elif method == 'weighted':
            sample = weighted_random_sample(n_sample, inter_items, item_list, cum_sums=item_cumulate_count)
        else:
            raise ValueError(f"invalid sampling method {method}")

        row.extend(sample)

        negative_sampled_test.append(row)

    return negative_sampled_test


def loading_data(data_type: str) -> Tuple[Any, Any, Any]:
    user_col = 'user_id'
    item_col = 'item_id'

    if data_type == '1M':
        file_path = os.path.join(CONFIG.DATA, 'movielens', 'ml-1m')
        loading_function = loading_movielens_1m
    else:
        raise ValueError(f"unknown data type {data_type}")

    return loading_function(file_path)
