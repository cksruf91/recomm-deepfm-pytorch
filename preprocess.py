import argparse
import os
import random
from itertools import accumulate
from typing import Tuple, Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from common.loading_functions import loading_data
from common.utils import progressbar, to_sparse_matrix
from config import CONFIG

random.seed(42)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='1M', choices=['10M', '1M', 'BRUNCH'], help='데이터셋', type=str)
    return parser.parse_args()


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


def train_test_split(df: DataFrame, user_col: str, time_col: str) -> Tuple[DataFrame, DataFrame]:
    """ 학습 테스트 데이터 분리 함수
    각 유저별 마지막 interaction 읕 테스트로 나머지를 학습 데이터셋으로 사용

    Args:
        df: 전체 데이터
        user_col: 기준 유저 아이디 컬럼명
        time_col: 기준 아이템 아이디 컬럼명

    Returns: 학습 데이터셋, 테스트 데이터셋
    """

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


def movielens_preprocess(interactions: DataFrame, items: DataFrame, users: DataFrame) -> Tuple[Any, Any, Any]:
    interactions.sort_values(['user_id', 'Timestamp'], inplace=True)

    # prev_item_id 생성
    interactions['prev_item_id'] = [-1] + interactions.item_id.tolist()[:-1]
    interactions.loc[interactions.user_id.diff(1) != 0, 'prev_item_id'] = -1
    interactions = interactions[interactions['prev_item_id'] != -1]

    # negative sampleing for train
    num_item = interactions.item_id.max()
    num_user = interactions.user_id.max()

    mat = to_sparse_matrix(interactions, num_user, num_item, 'user_id', 'item_id', 'Rating')
    temp_data = {
        'user_id': [], 'item_id': [], 'Timestamp': [], 'Rating': [], 'prev_item_id': []
    }

    negative_sample_size = 5
    length = len(interactions)

    for i, (uid, iid, timestamp, piid) in enumerate(zip(interactions['user_id'], interactions['item_id'],
                                                        interactions['Timestamp'], interactions['prev_item_id'])):
        progressbar(length, i + 1, suffix='generate negative samples')
        temp_data['user_id'].append(uid)
        temp_data['item_id'].append(iid)
        temp_data['Timestamp'].append(timestamp)
        temp_data['Rating'].append(1)
        temp_data['prev_item_id'].append(piid)

        for _ in range(negative_sample_size):
            j = np.random.randint(num_item)
            while mat.get((uid, j)):
                j = np.random.randint(num_item)
            temp_data['user_id'].append(uid)
            temp_data['item_id'].append(j)
            temp_data['Timestamp'].append(timestamp)
            temp_data['Rating'].append(0)
            temp_data['prev_item_id'].append(piid)

    interactions = pd.DataFrame(temp_data)

    # 장르 멀티 핫 인코딩
    items['Genres'] = items['Genres'].map(lambda x: set(x.split('|')))

    genres = set()
    for row in items['Genres']:
        genres.update(row)
    genres = {g: i for i, g in enumerate(genres)}

    multi_hot_encode = np.zeros([len(items), len(genres)])
    for i, item in enumerate(items['Genres']):
        for it in item:
            multi_hot_encode[i, genres[it]] = 1

    items = pd.concat(
        [items, pd.DataFrame(multi_hot_encode, columns=list(genres))], axis=1
    )

    interactions = interactions.merge(
        items[['item_id'] + list(genres)], on='item_id', how='left', validate='m:1'
    )

    # 학습 레이블 생성
    interactions['Rating'] = (interactions['Rating'] >= 1).astype(int)

    interactions = interactions[['item_id', 'user_id', 'Rating', 'Timestamp', 'prev_item_id'] + list(genres)]
    items = items[["MovieID", "Title", "Genres", "item_id"]]
    return interactions, items, users


def preprocess_data(data_type: str, interactions: DataFrame, items: DataFrame, users: DataFrame) -> Tuple[
    Any, Any, Any]:
    if data_type == '1M':
        loading_function = movielens_preprocess
    else:
        raise ValueError(f"unknown data type {data_type}")

    return loading_function(interactions, items, users)


if __name__ == '__main__':
    argument = args()

    log_data, item_meta, user_meta = loading_data(argument.dataset)
    log_data, item_meta, user_meta = preprocess_data(
        argument.dataset, log_data, item_meta, user_meta
    )

    train, test = train_test_split(log_data, user_col='user_id', time_col='Timestamp')
    assert len(test[test['Rating'] == 0]) == 0
    print(f'train data size : {len(train)}, test data size : {len(test)}')
    print(f'total item size : {len(item_meta)}, total user size : {len(user_meta)}')

    test_negative = get_negative_samples(
        train[train['Rating'] == 1], test, 'user_id', 'item_id', n_sample=99, method='random'
    )

    save_dir = os.path.join(CONFIG.DATA, argument.dataset)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    train.to_csv(os.path.join(save_dir, 'train.tsv'), sep='\t', index=False)
    test.to_csv(os.path.join(save_dir, 'test.tsv'), sep='\t', index=False)
    item_meta.to_csv(os.path.join(save_dir, 'item_meta.tsv'), sep='\t', index=False)
    user_meta.to_csv(os.path.join(save_dir, 'user_meta.tsv'), sep='\t', index=False)

    with open(os.path.join(save_dir, 'negative_test.dat'), 'w') as f:
        for row in test_negative:
            row = '\t'.join([str(v) for v in row])
            f.write(row + '\n')
