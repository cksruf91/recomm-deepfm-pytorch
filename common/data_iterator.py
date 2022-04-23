import torch
from torch.utils.data import Dataset


class Iterator(Dataset):

    def __init__(self, df, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.data = df[['user_id', 'prev_item_id', 'item_id']].values
        self.genres = df[
                ['Adventure', 'Crime', 'Animation', 'Western', 'Documentary', 'Mystery',
                 'Musical', 'Drama', 'Comedy', 'Fantasy', 'Horror', 'Action', 'Thriller',
                 'War', 'Film-Noir', 'Sci-Fi', 'Romance', 'Children\'s']
            ].values
        self.labels = df['Rating'].values

        self.data_tensor = self._to_tensor(self.data)
        self.genre_tensor = self._to_tensor(self.genres, dtype=torch.float32)
        self.label_tensor = self._to_tensor(self.labels, dtype=torch.float32)

    def _to_tensor(self, value, dtype=torch.int64):
        return torch.tensor(value, device=self.device, dtype=dtype)

    def __getitem__(self, index):
        # data = self._to_tensor(self.data[index])
        # label = self._to_tensor(self.labels[index], dtype=torch.float32)
        return self.data_tensor[index], self.genre_tensor[index], self.label_tensor[index]

    def __len__(self):
        return len(self.labels)


class TestIterator(Dataset):

    def __init__(self, test_df, test_file, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device

        self.prev_item_id = self._to_tensor(test_df[['prev_item_id']].values)
        self.genres = self._to_tensor(
            test_df[['Adventure', 'Crime', 'Animation', 'Western', 'Documentary', 'Mystery',
                     'Musical', 'Drama', 'Comedy', 'Fantasy', 'Horror', 'Action', 'Thriller',
                     'War', 'Film-Noir', 'Sci-Fi', 'Romance', 'Children\'s']].values,
            dtype=torch.float32
        )
        
        self.read_file(test_file)
        self.n_item = len(self.data[0][1:])

        self.label = self._to_tensor(self.label, dtype=torch.float32)
        self.data = self._to_tensor(self.data)
        

    def read_file(self, test_file):
        self.data = []
        self.label = []
        with open(test_file, 'r') as f:
            for row in f:
                row = [int(r) for r in row.split('\t')]
                self.data.append(row)
                label = [1] + [0] * (len(row[1:]) - 1)
                self.label.append(label)

    def _to_tensor(self, value, dtype=torch.int64):
        return torch.tensor(value, device=self.device, dtype=dtype)

    def __getitem__(self, index):
        user = self.data[index][0].repeat(self.n_item)
        pitem = self.prev_item_id[index][0].repeat(self.n_item)
        item = self.data[index][1:]
        genres = self.genres[index].repeat(self.n_item, 1)
        return torch.stack([user, pitem, item], dim=1), genres, self.label[index]

    def __len__(self):
        return len(self.data)
