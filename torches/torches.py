#! /usr/bin/env python3

from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch.nn import Sequential

from tools.losses import Objective
from tools.errors import NotBuiltError


class TorchModel(ABC, torch.nn.Module):
    def __init__(
        self, model_pieces, *args,
        model_name="Default", **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.pieces = Sequential()
        for key in model_pieces:
            self.pieces.add_module(
                key, Sequential(model_pieces[key])
            )

        self.model_name = model_name
        self.use_cuda = False
        self.loss = None
        self.tensorboard_writer = None
        self.built = False

    def build(
        self, loss_func_dict, optimizer=torch.optim.Adam,
        learning_rate=1e-3, try_cuda=True, cuda_device='0'
    ):

        if try_cuda:
            self.use_cuda = torch.cuda.is_available()

        self.device = torch.device(
            f"cuda:{cuda_device}" if self.use_cuda else "cpu"
        )

        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.backends.cudnn.benchmark = True

        self.loss_function = Objective(loss_func_dict)

        self.opt = optimizer(self.parameters(), lr=learning_rate)

        self.to(self.device)
        self.built = True

    def forward(self, inputs):
        if not(self.built):
            msg = "Run build method on model before first call!"
            raise NotBuiltError(msg)

        pieces = OrderedDict()
        current_inputs = inputs

        for key in self.pieces._modules:
            pieces[key] = self.pieces._modules[key](current_inputs)
            current_inputs = pieces[key]

        return pieces

    @abstractmethod
    def fit_batch(self):
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        raise NotImplementedError


class Autoencoder(TorchModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit_batch(self, batch):
        pass

    def fit(self, inputs):
        pass
