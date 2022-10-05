import os

import torch

from fastai.callback.core import Callback
from fastai.callback.tracker import TrackerCallback


class SaveStatedictCallback(TrackerCallback):
    _only_train_loop, order = True, TrackerCallback.order + 1

    def __init__(self, path='', monitor='valid_loss', comp=None, min_delta=0., fname='model', reset_on_fit=True):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
        self.fname = fname
        self.path = path

    def after_epoch(self):
        super().after_epoch()
        if self.new_best:
            print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
            torch.save(self.learn.model.state_dict(), os.path.join(self.path, self.fname + '.statedict'))


class SaveStatedictEveryNEpochCallback(Callback):
    _only_train_loop, order = True, TrackerCallback.order + 1

    def __init__(self, n_to_save=1, path='', fname='model'):
        super().__init__()
        self.fname = fname
        self.path = path
        self.n_to_save = n_to_save

    def after_epoch(self):
        if self.epoch % self.n_to_save == 0:
            print(f'Saving at epoch {self.epoch}')
            torch.save(self.learn.model.state_dict(), os.path.join(self.path, f'{self.fname}-{self.epoch}.statedict'))
