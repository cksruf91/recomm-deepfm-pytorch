# recomm-deepfm-pytorch

___work in progress___
## Dataset
* [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
```bash
cd {project Dir}/recomm-hrnn-pytorch/datasets/movielens
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```

* [MovieLens 10M Dataset](https://grouplens.org/datasets/movielens/10m/)
```bash
cd {project Dir}/recomm-hrnn-pytorch/datasets/movielens
wget https://files.grouplens.org/datasets/movielens/ml-10m.zip
unzip ml-10m.zip
```

## data preprocess
```shell
python preprocess.py -d 1M
```

## train model
```shell
python deepfm_train.py -d 1M -v 0.1.0 -k 10 -ed 16 -lr 0.001 -bs 256 -ly 64 32 -o 0.2
```
