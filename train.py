import argparse
import os
from typing import Callable

import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, Adagrad, Adadelta
from torch.utils.data import DataLoader

from common.data_iterator import to_sparse_matrix, Iterator, TestIterator
from config import CONFIG
from model.callbacks import ModelCheckPoint, MlflowLogger
from model.metrics import nDCG, RecallAtK
from model.deepfm_model import DeepFM


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='1M', choices=['10M', '1M', 'BRUNCH'], help='데이터셋', type=str)
    parser.add_argument('-v', '--model_version', required=True, help='모델 버전', type=str)
    parser.add_argument('-k', '--eval_k', default=10, help='', type=int)
    parser.add_argument('-f', '--factor', default=8, help='embedding size', type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, help='learning rate', type=float)
    parser.add_argument('-bs', '--batch_size', default=256, help='batch size', type=int)
    parser.add_argument('-ns', '--negative_size', default=4, help='train negative sample size', type=int)
    parser.add_argument('-ly', '--layers', default=[64, 32, 16, 8], help='mlp layer size', type=int, nargs='+')
    parser.add_argument('-wd', '--weight_decay', default=0., help='weight decay for optimizer', type=float)
    parser.add_argument('-o', '--dropout', default=0., help='dropout ratio', type=float)
    parser.add_argument('-cpu', '--cpu',  action='store_true', help='')

    return parser.parse_args()


def get_optimizer(model, name: str, lr: float, wd: float = 0.) -> Callable:
    """ get optimizer
    Args:
        model: pytorch model
        name: optimizer name
        lr: learning rate
        wd: weight_decay(l2 regularization)

    Returns: pytorch optimizer function
    """

    functions = {
        'Adagrad': Adagrad(model.parameters(), lr=lr, eps=0.00001, weight_decay=wd),
        'Adadelta': Adadelta(model.parameters(), lr=lr, eps=1e-06, weight_decay=wd),
        'Adam': Adam(model.parameters(), lr=lr, weight_decay=wd)
    }
    try:
        return functions[name]
    except KeyError:
        raise ValueError(f'optimizer [{name}] not exist, available optimizer {list(functions.keys())}')


if __name__ == '__main__':
    argument = args()

    save_dir = os.path.join(CONFIG.DATA, argument.dataset)
    train_data = pd.read_csv(os.path.join(save_dir, 'train.tsv'), sep='\t')
    item_meta = pd.read_csv(os.path.join(save_dir, 'item_meta.tsv'), sep='\t', low_memory=False)
    user_meta = pd.read_csv(os.path.join(save_dir, 'user_meta.tsv'), sep='\t')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if argument.cpu:
        device = torch.device('cpu')

    matrix = to_sparse_matrix(train_data, int(user_meta.user_id.max()), int(item_meta.item_id.max()),
                              'user_id', 'item_id', 'Rating')
    train_iterator = Iterator(matrix, argument.negative_size, device=device)
    train_dataloader = DataLoader(train_iterator, batch_size=argument.batch_size, shuffle=True)

    test_iterator = TestIterator(os.path.join(save_dir, 'negative_test.dat'), device=device)
    test_dataloader = DataLoader(test_iterator, batch_size=1, shuffle=False)

    n_user, n_item = matrix.shape

    model_params = {
        'Factor': argument.factor,  # 500
        'learningRate': argument.learning_rate,
        'loss': 'BCEWithLogitsLoss',
        'optimizer': 'Adam',
        'k': argument.eval_k,
        'batchSize': argument.batch_size,
        'negative_size': argument.negative_size,
        'layers': argument.layers,
        'num_users': n_user, 'num_items': n_item,
        'weight_decay': argument.weight_decay,
        'component': ['mlp', 'gmf'],
        'dropout': argument.dropout
    }

    nmf = DeepFM(n_user, n_item, n_factor=argument.factor, layers=argument.layers, dropout=argument.dropout,
                   component=model_params['component'], device=device)

    cross_entropy = BCEWithLogitsLoss()
    optim = get_optimizer(nmf, name=model_params['optimizer'], lr=argument.learning_rate, wd=argument.weight_decay)

    metrics = [nDCG(), RecallAtK()]
    model_version = f'nmf_v{argument.model_version}'
    callback = [
        ModelCheckPoint(os.path.join(
            'result', argument.dataset, model_version + '-e{epoch:02d}-loss{val_loss:1.3f}-nDCG{val_nDCG:1.3f}.zip'),
            monitor='val_nDCG', mode='max'
        ),
        MlflowLogger(experiment_name=argument.dataset, model_params=model_params, run_name=model_version,
                     log_model=False)
    ]
    print(nmf)
    print(f"device : {device}")
    nmf.fit(50, train_dataloader, test_dataloader, loss_func=cross_entropy, optimizer=optim, metrics=metrics,
            callback=callback)
