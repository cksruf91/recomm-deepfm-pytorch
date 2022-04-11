import argparse
import os

import pandas as pd
import torch
from torch.nn import Sigmoid

from config import CONFIG
from model.ncf_model import NeuralMF


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', required=True, help='확인할 유저 번호', type=int)
    parser.add_argument('-d', '--dataset', default='10M', choices=['10M', '1M', 'BRUNCH'], help='데이터셋', type=str)
    parser.add_argument('-w', '--weight', required=True, help='모델 가중치값', type=str)
    parser.add_argument('-k', '--eval_k', default=25, help='', type=int)
    return parser.parse_args()


def get_user_test_data(test_data, user_id):
    with open(test_data, 'r') as file:
        for line in file:
            line = [int(l) for l in line.split('\t')]
            if line[0] == user_id:
                return line[1], line[2:]
    raise ValueError(f'User {user_id} is not exist')


def prediction(model, user, items):
    sigmoid = Sigmoid()
    with torch.no_grad():
        user = [user for _ in range(len(items))]
        scores = torch.zeros(len(items), device=device)
        for i, (it, ur) in enumerate(zip(items, user)):
            it = torch.tensor(it, device=device, dtype=torch.int64)
            ur = torch.tensor(ur, device=device, dtype=torch.int64)
            scores[i] = sigmoid(model(ur, it))
        _, indices = torch.topk(scores, k=argument.eval_k)

    return [items[i] for i in indices.cpu().numpy()]


if __name__ == '__main__':
    argument = args()

    # loading data
    save_dir = os.path.join(CONFIG.DATA, argument.dataset)
    train_data = pd.read_csv(os.path.join(save_dir, 'train.tsv'), sep='\t')
    item_meta = pd.read_csv(os.path.join(save_dir, 'item_meta.tsv'), sep='\t', low_memory=False)
    user_meta = pd.read_csv(os.path.join(save_dir, 'user_meta.tsv'), sep='\t')
    n_user = int(user_meta.user_id.max() + 2)  # 30606
    n_item = int(item_meta.item_id.max() + 2)  # 235788
    positive_item, negative_item = get_user_test_data(os.path.join(save_dir, 'negative_test.dat'), argument.user)

    # loading model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    nmf = NeuralMF(n_user, n_item, n_factor=8, layers=[64, 32, 16, 8], component=['gmf', 'mlp'], device=device)
    nmf.load(os.path.join('result', argument.dataset, argument.weight))
    nmf.eval()
    
    display_cols = ['Title', 'Genres']
    if argument.dataset == 'BRUNCH':
        display_cols = ['title', 'keyword_list', 'display_url']

    # user interaction items
    train_data = train_data[train_data['user_id'] == argument.user]
    train_data = train_data.merge(item_meta[['item_id'] + display_cols], on='item_id', how='left')
    print(train_data)

    # predictions
    recommend_items = prediction(nmf, argument.user, list(range(n_item)))  # [positive_item] + negative_item
    recommend_items = pd.DataFrame({'item_id': recommend_items, 'user_id': argument.user})
    recommend_items = recommend_items.merge(
        item_meta[['item_id'] + display_cols], on='item_id', how='left'
    )
    recommend_items = recommend_items.merge(
        user_meta, on='user_id', how='left', validate='m:1'
    )
    print(recommend_items)
