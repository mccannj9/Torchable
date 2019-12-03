#! /usr/bin/env python3

import numpy as np

from torch.utils.data import Dataset

from utensils.mnist import mnist_image_to_numpy
from utensils.mnist import mnist_label_to_numpy
from utensils.errors import TooManyTimestepsError


class MnistDataset(Dataset):

    def __init__(
        self, images_file, labels_file, dtype=np.float32,
        max_images_to_load=None, onehot_encode=True, shape=None
    ):
        self.images = mnist_image_to_numpy(
            images_file, dtype=dtype,
            max_images_to_load=max_images_to_load
        )

        self.labels = mnist_label_to_numpy(
            labels_file, max_images_to_load=max_images_to_load,
            onehot_encode=onehot_encode, dtype=dtype
        )

        if shape:
            self.images = self.images.reshape(
                self.images.shape[0], *shape
            )

        self.images /= 255

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class NflRushingDataset(Dataset):
    def __init__(
        self, images_file, labels_file, dtype=np.float32, shape=None
    ):
        self.images = np.load(images_file).astype(dtype)
        self.labels = np.load(labels_file).astype(dtype)

        if shape:
            self.images = self.images.reshape(
                self.images.shape[0], *shape
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class CreditCardDataset(Dataset):
    def __init__(
        self, data_file, labels_file=None,
        amounts_file=None, dtype=np.float32
    ):
        self.data = np.load(data_file).astype(dtype)

        if not(labels_file):
            self.labels = self.data[:, -1]
            self.amounts = self.data[:, -2]
            self.data = self.data[:, :-2]

        else:
            self.labels = np.load(labels_file).astype(dtype)
            self.amounts = np.load(amounts_file).astype(dtype)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.amounts[idx], self.labels[idx]


class ElectricityGrid(Dataset):
    def __init__(
        self, data_file, metadata_file=None,
        timesteps=None, dtype=np.float32
    ):
        if timesteps:
            self.timesteps = timesteps
        else:
            self.timesteps = 1
        self.data = np.load(data_file).astype(dtype)
        self.data /= self.data.max(axis=0)

    def __len__(self):
        return len(self.data) - self.timesteps - 1

    def __getitem__(self, idx):
        return (
            self.data[idx: idx+self.timesteps, :],
            self.data[idx+1:idx+1+self.timesteps, :]
        )


class ElectricGridPredict(Dataset):
    def __init__(
        self, latent_with_meta, original_data_file,
        embedded_timesteps=50, timesteps=1, dtype=np.float32
    ):

        # don't set timesteps to 0!
        if timesteps < 1:
            self.timesteps = 1
        else:
            self.timesteps = timesteps
        self.embedded_timesteps = embedded_timesteps
        self.latent_w_meta = np.load(latent_with_meta).astype(dtype)
        self.targets = np.load(original_data_file).astype(dtype)
        self.targets /= self.targets.max(axis=0)

        max_ts = self.targets.shape[0] - self.latent_w_meta.shape[0]

        if self.timesteps > max_ts:
            msg = "Too many timesteps for forecast training"
            raise TooManyTimestepsError(msg)

    def __len__(self):
        return len(self.latent_w_meta)

    def __getitem__(self, idx):
        y = self.embedded_timesteps
        x = self.timesteps
        return (
            self.latent_w_meta[idx, :],
            self.targets[idx+y:idx+x+y, :].squeeze()
        )
