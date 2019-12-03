#! /usr/bin/env python3

import torch
from torch.nn import Module


class LstmAllHidden(Module):
    def __init__(
        self, input_size, hidden_size, *args,
        num_layers=1, bias=True, batch_first=True,
        dropout=0.0, bidirectional=False, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            bias=bias, batch_first=batch_first, dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self, inputs):
        o, (h, c) = self.lstm(inputs)
        return o


class LstmCellOnly(Module):
    def __init__(
        self, input_size, hidden_size, *args,
        num_layers=1, bias=True, batch_first=True,
        dropout=0.0, bidirectional=False, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            bias=bias, batch_first=batch_first, dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self, inputs):
        o, (h, c) = self.lstm(inputs)
        return c.squeeze()


class ExpandAndRepeatOutput(Module):
    def __init__(self, axis, repeat_times):
        super().__init__()
        self.expanding_axis = axis
        self.reps = repeat_times
        self.expand = [-1, -1, -1]
        self.expand[self.expanding_axis] *= -self.reps

    def forward(self, inputs):
        return inputs.unsqueeze(
            self.expanding_axis
        ).expand(*self.expand)

    def extra_repr(self):
        return f"expand={self.expanding_axis}, reps={self.reps}"
