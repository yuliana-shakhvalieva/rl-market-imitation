import random
import numpy as np
import pandas as pd
from utils import round_price
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DiscriminatorDataset(Dataset):
    def __init__(self, glass_len):
        self.slice_buy = pd.read_csv('./SBER_slice_buy.txt')
        self.slice_sell = pd.read_csv('./SBER_slice_sell.txt')
        self.glass_len = glass_len

    def __len__(self):
        return len(self.slice_buy) // self.glass_len

    def __getitem__(self, idx):
        buy = np.array(self.slice_buy[idx:idx + self.glass_len])
        sell = np.array(self.slice_sell[idx:idx + self.glass_len])

        sell.sort(axis=0)
        buy[::-1].sort(axis=0)

        return np.vstack((buy, sell)), 0


class DiscriminatorModel(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(nn.BatchNorm1d(int(state_dim/2)),
                                   nn.Flatten(),
                                   nn.Linear(state_dim, 128),
                                   nn.ELU(),
                                   nn.Linear(128, 1),
                                   nn.Sigmoid()).to(device)

    def forward(self, state):
        state = state.to(torch.float32)
        eps = torch.rand(*state.shape, dtype=torch.float32).to(device)
        return self.model(state + eps)


class Discriminator:
    def __init__(self, glass_len):
        self.glass_len = glass_len
        self.state_dim = (self.glass_len * 2) * 2

        self.train_dataset = DiscriminatorDataset(glass_len)
        train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True)
        self.train_loader_iterator = iter(train_loader)
        self.discriminator = DiscriminatorModel(self.state_dim)
        self.loss_f = nn.BCELoss()
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-3)

    def get_reward_from_discriminator(self, glass_slice):
        X, y, glass_position = self.get_batch(round_price(glass_slice))
        self.discriminator.zero_grad()
        prediction = self.discriminator.forward(X.to(device))
        loss = self.loss_f(prediction.squeeze(), y.to(torch.float32).to(device))
        loss.backward()
        self.optimizer.step()
        return 10 * (2 * prediction.detach().cpu().numpy()[glass_position] - 1)

    def get_batch(self, glass_slice):
        try:
            X, y = next(self.train_loader_iterator)
        except StopIteration:
            train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True)
            self.train_loader_iterator = iter(train_loader)
            X, y = next(self.train_loader_iterator)
        glass_position = random.randint(0, len(y)-1)
        X[glass_position] = torch.tensor(glass_slice)
        y[glass_position] = torch.tensor(1)
        return X, y, glass_position
