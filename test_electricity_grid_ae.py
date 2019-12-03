#! /usr/bin/env python3

from collections import OrderedDict

from tools.datasets import ElectricityGrid
from tools.special import LstmAllHidden, LstmCellOnly
from tools.special import ExpandAndRepeatOutput

from torches.torches import Autoencoder

import torch

mse = torch.nn.MSELoss(reduction='sum')

bs = 256
shuf = True
nepochs = 30
ts = 50

data_home = "/home/jamc/.datasets/electricity_grid"
data_fn = f"{data_home}/LD2011_2014_.npy"


dataset = ElectricityGrid(data_fn, timesteps=ts)

# scaling for tanh input
dataset.data = 2 * dataset.data - 1.0

training = torch.utils.data.DataLoader(
    dataset, batch_size=bs, shuffle=shuf
)

num_clients = dataset.data.shape[-1]
inter_dim = 128
latent_dim = 16

encoder = OrderedDict(
    (
        ('lstm_enc_1', LstmAllHidden(
            num_clients, inter_dim, batch_first=True
        )),
        ('lstm_enc_2', LstmCellOnly(
            inter_dim, latent_dim, batch_first=True
        )),
    )
)

decoder = OrderedDict(
    (
        ('repeater', ExpandAndRepeatOutput(1, ts)),
        ('lstm_dec_1', LstmAllHidden(
            latent_dim, inter_dim, batch_first=True
        )),
        ('lstm_enc_2', LstmAllHidden(
            inter_dim, num_clients, batch_first=True
        ))
    )
)

model_dict = {
    "encoder": encoder, "decoder": decoder
}

model = Autoencoder(
    model_dict, model_name="Autoencode_Egrid"
)

model.build({'mse': mse}, try_cuda=True)

model.fit(training, nepochs=nepochs)
