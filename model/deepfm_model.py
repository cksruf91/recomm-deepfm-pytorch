import numpy as np
import torch


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)
    

class MultiHotEmbedding(torch.nn.Module):
    
    def __init__(self, multi_hotencoding_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.emb_w = torch.nn.Parameter(torch.zeros([multi_hotencoding_size, embed_dim], dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.emb_w)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, multi_hotencoding_size)``
        
        return (batch_size, embed_dim)
        """
        return torch.matmul(x, self.emb_w).reshape(-1, 1, self.embed_dim)


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


# class DeepFactorizationMachineModel(torch.nn.Module):
#     """
#     A pytorch implementation of DeepFM.

#     Reference:
#         H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
#     """

#     def __init__(self, field_dims, embed_dim, mlp_dims, dropout, device):
#         super().__init__()
#         self.linear = FeaturesLinear(field_dims)
#         self.fm = FactorizationMachine(reduce_sum=True)
#         self.embedding = FeaturesEmbedding(field_dims, embed_dim)
#         self.embed_output_dim = len(field_dims) * embed_dim
#         self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

#         self.to(device)

#     def forward(self, x):
#         """
#         :param x: Long tensor of size ``(batch_size, num_fields)``
#         """
#         embed_x = self.embedding(x)  # [batch_size, num_fields, emb_size] <-[batch_size, num_fields]
#         x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
#         return torch.sigmoid(x.squeeze(1))

    
class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, multi_hot_size, embed_dim, mlp_dims, dropout, device):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.multi_embedding = MultiHotEmbedding(multi_hot_size, embed_dim)
        self.embed_output_dim = (len(field_dims) + 1) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        self.to(device)

    def forward(self, x, genres):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)  # [batch_size, num_fields, emb_size] <-[batch_size, num_fields]
        embed_genres = self.multi_embedding(genres)
        embed_x = torch.concat([embed_x, embed_genres], dim=1)
        
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))