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

        self.obj = Objective(loss_func_dict)

        self.opt = optimizer(self.parameters(), lr=learning_rate)

        self.to(self.device)
        self.built = True

    def forward(self, inputs):
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
        if not(self.built):
            msg = "Run build method on model before first call!"
            raise NotBuiltError(msg)

        self.opt.zero_grad()

        # get outputs, standard autoencoder has only one
        outputs = self.forward(batch)
        keys = list(outputs.keys())
        predictions = outputs[keys[-1]]

        # reconstruction loss with batch as target
        loss_params = ((predictions, batch),)
        self.loss = self.obj(loss_params)
        self.loss.backward()
        self.opt.step()
        return predictions

    def fit(
        self, inputs, nepochs=1, logint=100,
        continue_counter=0, continue_epoch=0
    ):

        counter = continue_counter
        nepochs += continue_epoch

        self.train()
        for i in range(continue_epoch, nepochs):
            # assume inputs enum returns batch, target pair
            # discard target for autoencoders, reconstruct batch
            for j, (batch, _) in enumerate(inputs):
                _ = self.fit_batch(batch)

                if j % logint == 0:
                    outmsg = f"[Epoch {i}, Batch {j} Loss] "
                    outmsg += f"Total: {round(self.loss.item(), 3)}"
                    for k in self.obj.losses:
                        outmsg += f" {k}: " + str(
                            round(self.obj.losses[k].item(), 3)
                        )
                    print(outmsg)
                counter += 1
