import argparse
import os
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.optim import Adagrad, Adadelta, Adam
from torch.utils.data import DataLoader

from common.data_iterator import Iterator, TestIterator
from config import CONFIG
from model.callbacks import ModelCheckPoint, MlflowLogger
from model.deepfm_model import DeepFactorizationMachineModel
from model.metrics import nDCG, RecallAtK


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='1M', choices=['10M', '1M', 'BRUNCH'], help='데이터셋', type=str)
    parser.add_argument('-v', '--model_version', required=True, help='모델 버전', type=str)
    parser.add_argument('-k', '--eval_k', default=10, help='', type=int)
    parser.add_argument('-ed', '--embed_dim', default=16, help='embedding size', type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, help='learning rate', type=float)
    parser.add_argument('-bs', '--batch_size', default=256, help='batch size', type=int)
    parser.add_argument('-ly', '--layers', default=[64, 32], help='mlp layer size', type=int, nargs='+')
    parser.add_argument('-o', '--dropout', default=0.2, help='dropout ratio', type=float)
    parser.add_argument('-cpu', '--cpu', action='store_true', help='')

    return parser.parse_args()


def get_optimizer(model, name: str, lr: float, wd: float = 0.) -> Callable:
    """ get optimizer
    Args:
        model: pytorch model
        name: optimizer name
        lr: learning rate
        wd: weight_decay(l2 regulraization)

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


def train_progressbar(total: int, i: int, bar_length: int = 50, prefix: str = '', suffix: str = '') -> None:
    """progressbar
    """
    dot_num = int((i + 1) / total * bar_length)
    dot = '■' * dot_num
    empty = ' ' * (bar_length - dot_num)
    sys.stdout.write(f'\r {prefix} [{dot}{empty}] {i / total * 100:3.2f}% {suffix}')


def train(model, epoch, train_dataloader, test_dataloader, loss_func, optim, metrics=[], callback=[]):
    for e in range(epoch):
        # ------ train --------
        model.train()

        start_epoch_time = time.time()
        train_loss = 0
        total_step = len(train_dataloader)

        history = {}

        for step, (data, genres, labels) in enumerate(train_dataloader):
            # ------ step start ------
            if ((step + 1) % 50 == 0) | (step + 1 >= total_step):
                train_progressbar(
                    total_step, step + 1, bar_length=30,
                    prefix=f'train {e + 1:03d}/{epoch} epoch', suffix=f'{time.time() - start_epoch_time:0.2f} sec '
                )
            pred = model(data, genres)
            loss = loss_func(pred, labels)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            model.zero_grad()

            train_loss += loss.item()

            if step >= total_step:
                break
            # ------ step end ------

        history['epoch'] = e + 1
        history['time'] = np.round(time.time() - start_epoch_time, 2)
        history['train_loss'] = train_loss / total_step

        sys.stdout.write(f"loss : {history['train_loss']:3.3f}")

        # ------ test  --------
        model.eval()
        val_loss = 0
        y_pred, y_true = [], []
        with torch.no_grad():
            for step, (data, genres, labels) in enumerate(test_dataloader):

                # random shuffle
                idx = torch.randperm(len(data))
                data = data[idx]
                genres = genres[idx]
                labels = labels[idx]

                pred = model(data, genres)
                loss = loss_func(pred, labels)
                val_loss += loss.item()

                _, indices = torch.topk(pred, k=10)
                y_pred.append(data[indices][:, 2].cpu().tolist())
                y_true.append(data[labels == 1][0, 2].item())

        history['val_loss'] = val_loss / step
        result = f" val_loss : {history['val_loss']:3.3f}"

        for func in metrics:
            metrics_value = func(y_pred, y_true)
            history[f'{func}'] = metrics_value
            result += f' val_{func} : {metrics_value:3.3f}'

        for func in callback:
            func(model, history)

        print(
            f" val_loss : {history['val_loss']:3.3f}, nDCG : {history['nDCG']:3.3f}, recall : {history['recallAtK']:3.3f} ")


if __name__ == '__main__':
    argument = args()

    save_dir = os.path.join(CONFIG.DATA, argument.dataset)
    train_data = pd.read_csv(os.path.join(save_dir, 'train.tsv'), sep='\t')
    test_data = pd.read_csv(os.path.join(save_dir, 'test.tsv'), sep='\t')
    item_meta = pd.read_csv(os.path.join(save_dir, 'item_meta.tsv'), sep='\t', low_memory=False)
    user_meta = pd.read_csv(os.path.join(save_dir, 'user_meta.tsv'), sep='\t')
    
    genres_columns = ['Adventure', 'Crime', 'Animation', 'Western', 'Documentary', 'Mystery',
                     'Musical', 'Drama', 'Comedy', 'Fantasy', 'Horror', 'Action', 'Thriller',
                     'War', 'Film-Noir', 'Sci-Fi', 'Romance', 'Children\'s']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if argument.cpu:
        device = torch.device('cpu')

    model_params = {
        'learningRate': argument.learning_rate,
        'loss': 'BCELoss',
        'optimizer': 'Adam',
        'k': argument.eval_k,
        'batchSize': argument.batch_size,
        'negative_size': 5,
        'num_users': train_data['user_id'].nunique(), 'num_items': train_data['item_id'].nunique()
    }

    train_iterator = Iterator(train_data, device=device)
    train_dataloader = DataLoader(train_iterator, batch_size=argument.batch_size, shuffle=True)
    test_iterator = TestIterator(test_data, os.path.join(save_dir, 'negative_test.dat'), device=device)
    # test_dataloader = DataLoader(test_iterator, batch_size=1, shuffle=False)

    field_dims = np.max(train_iterator.data, axis=0) + 1

    # item_id 와 prev_item_id 가 embedding parameter를 공유하도록 한다.
    field_dims[1] = 0  # array([6040, 3883, 3883]) -> array([6040, 0, 3883])

    model = DeepFactorizationMachineModel(
        field_dims, multi_hot_size=len(genres_columns), 
        embed_dim=argument.embed_dim, mlp_dims=argument.layers, dropout=argument.dropout, device=device
    )
    print(model)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    print(f'model size : {param_size / 1024 / 1024:1.5f} mb')

    loss_func = torch.nn.BCELoss()
    optim = get_optimizer(model, model_params['optimizer'], model_params['learningRate'])
    metrics = [nDCG(), RecallAtK()]

    model_version = f'deepfm_v{argument.model_version}'
    callback = [
        # ModelCheckPoint(os.path.join(
        #     'result', argument.dataset,
        #     model_version + '-e{epoch:02d}-loss{val_loss:1.3f}-nDCG{val_nDCG:1.3f}.zip'),
        #     monitor='val_nDCG', mode='max'
        # ),
        # MlflowLogger(experiment_name=argument.dataset, model_params=model_params, run_name=model_version,
        #              log_model=False)
    ]

    train(model, 50, train_dataloader, test_iterator, loss_func, optim, metrics=metrics, callback=callback)
