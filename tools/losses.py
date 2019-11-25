#! /usr/bin/env python3

from collections import OrderedDict


class Objective(object):
    """
        A pickle-able version of a loss function that
        can be saved with a torch model object.

        This is a fix for a loss function generator
        which generated the loss function locally, but
        unfortunately is not able to be pickled.

        :param function_dict: a python (ordered) dict of
        function names (keys) and functions (values),
        which each take a tuple of two tensors as input,
        e.g. (prediction, target).
        :type function_dict: dict or OrderedDict

        ** Usage Example on Torch Modules **
        MSE = torch.nn.MSELoss(reduction='mean')
        KL = torch.distributions.kl_divergence
        loss_functions = {
            'mean_squared_error': MSE,
            'kl_divergence': KL
        }

        loss_func = Objective(loss_functions)

        # call it like this
        loss = loss_func(((recon, target), (post, prior)))

        # then for backpropagation
        loss.backward()
        optimizer.step()

    """

    def __init__(self, function_dict):
        self.funcs = function_dict
        self.losses = None

    def __call__(self, inputs):
        self.losses = OrderedDict()

        for f, (i, o) in zip(self.funcs.keys(), inputs):
            self.losses[f] = self.funcs[f](i, o)

        return sum(self.losses.values())
