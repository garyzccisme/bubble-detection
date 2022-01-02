import copy

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim.lr_scheduler as lr
from torch.utils.data import DataLoader, TensorDataset

from model.eval import accuracy


class HistEncoder(nn.Module):
    def __init__(self, input_channel, mid_channel, out_channel, dropout):
        super().__init__()
        self.lstm1 = nn.LSTM(input_channel, mid_channel, dropout=dropout, batch_first=True)
        self.lstm2 = nn.LSTM(mid_channel, out_channel, dropout=dropout, batch_first=True)
        self.activation = nn.ReLU()

    def forward(self, x):
        x, (_, _) = self.lstm1(x)
        _, (x, _) = self.lstm2(x)
        x = nn.BatchNorm1d(num_features=x.shape[2])(x[-1])
        x = self.activation(x)
        return x


class AggEncoder(nn.Module):
    def __init__(self, input_channel, mid_channel, out_channel, dropout):
        super().__init__()
        self.dense1 = nn.Linear(input_channel, mid_channel)
        self.dense2 = nn.Linear(mid_channel, out_channel)
        self.bn1 = nn.BatchNorm1d(mid_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_channel, mid_channel, out_channel, dropout):
        super().__init__()
        self.dense1 = nn.Linear(input_channel, mid_channel)
        self.dense2 = nn.Linear(mid_channel, out_channel)
        self.bn1 = nn.BatchNorm1d(mid_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.sigmoid(x)
        return x


class DoubleEncoderNet(nn.Module):
    def __init__(self, x1_dim, x2_dim, dropout):
        super().__init__()
        self.encoder1 = HistEncoder(x1_dim, 16, 8, dropout)
        self.encoder2 = AggEncoder(x2_dim, 16, 8, dropout)
        self.decoder = Decoder(16, 4, 1, dropout)

    def forward(self, x1, x2):
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x = self.decoder(x1, x2)
        return x[:, 0]


class WeightedBCELoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = [1, 1]

        self.weights = torch.tensor([2 * w / sum(weights) for w in weights])
        self.loss = nn.BCELoss(reduce=False)

    def forward(self, pred, true):
        weight_ = self.weights[true.data.view(-1).long()].view_as(true)
        loss = self.loss(pred, true)
        loss_class_weighted = loss * weight_
        loss_class_weighted = loss_class_weighted.mean()
        return loss_class_weighted


class DoubleEncoderModel:
    def __init__(self, x1_dim, x2_dim, dropout):
        self.network = DoubleEncoderNet(x1_dim, x2_dim, dropout)
        self.optimizer = Adam(self.network.parameters(), lr=0.1, weight_decay=0.1)
        self.scheduler = lr.StepLR(self.optimizer, step_size=1, gamma=0.99)
        # self.loss = nn.BCELoss(reduction='mean')
        self.loss = WeightedBCELoss(weights=[2, 3])
        # self.loss = TSBinaryLoss()
        self.dataloader = None

        self.best_loss = float('inf')
        self.best_acc = 0
        self.best_params = None

        # self.train_loss = []
        # self.train_acc = []

    def get_dataloader(self, x1, x2, y):
        x1 = torch.tensor(x1).float()
        x2 = torch.tensor(x2).float()
        y = torch.tensor(y).float()
        dataset = TensorDataset(x1, x2, y)
        self.dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    def fit(self, x1, x2, y, max_epoch=20):

        self.get_dataloader(x1, x2, y)
        epoch = 1
        while epoch <= max_epoch:
            print(f"=== Training epoch: {epoch}")
            self._fit()
            loss, acc = self.get_metric()
            print(f"--- Training Loss: {loss}, Training Accuracy: {acc}")

            if loss < self.best_loss:
                self.best_params = copy.deepcopy(self.network.state_dict())
                self.best_loss = loss

            # if acc > self.best_acc:
            #     self.best_params = copy.deepcopy(self.network.state_dict())
            #     self.best_acc = acc

            epoch += 1

        print(f"Training Finish, best loss is {self.best_loss}")
        # print(f"Training Finish, best accuracy is {self.best_acc}")

        self.network.load_state_dict(self.best_params)

    def _fit(self):
        self.network.train()
        for x1, x2, y in self.dataloader:
            pred = self.network(x1, x2)
            loss = self.loss(pred, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    def get_metric(self):
        self.network.eval()
        x1, x2, y = self.dataloader.dataset.tensors
        pred = self.network.forward(x1, x2)
        loss = self.loss(pred, y)
        acc = accuracy(pred.detach().numpy(), y.detach().numpy())
        return loss, acc

    def predict(self, x1, x2):
        self.network.eval()
        x1 = torch.tensor(x1).float()
        x2 = torch.tensor(x2).float()
        output = self.network.forward(x1, x2).detach().numpy()
        return output
