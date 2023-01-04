import pandas as pd
from torch.utils.data import Dataset
import torch


class MusicalFeaturesDataset(Dataset):
    def __init__(self, features: torch.Tensor, genres: torch.Tensor):
        super().__init__()

        assert features.shape[0] == genres.shape[0]
        assert len(genres.shape) == 1

        self.features = features
        self.genres = genres

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index, :], self.genres[index]