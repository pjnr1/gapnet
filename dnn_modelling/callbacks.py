import os

import torch

from fastai.callback.core import Callback
from fastai.callback.tracker import TrackerCallback


class SaveStatedictCallback(TrackerCallback):
    """
    Callback used with fastai when training the neural-network models for saving the statedict of the model after each
    epoch, given the model improved over the best model so far.
    """
    _only_train_loop = True
    order = TrackerCallback.order + 1

    def __init__(self, path='', monitor='valid_loss', comp=None, min_delta=0., fname='model', reset_on_fit=True):
        """
        Creates a SaveStatedictCallback object.

        Output path is::
            os.path.join(path, fname + '.statedict')

        @arg path:
            Folder to save the statedict in
        @arg monitor:
            What parameter to monitor
        @arg comp:
            See fastai.callback.tracker.TrackerCallback for details
        @arg min_delta:
            See fastai.callback.tracker.TrackerCallback for details
        @arg fname:
            Filename to use
        @arg reset_on_fit:
            See fastai.callback.tracker.TrackerCallback for details
        """
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
        self.fname = fname
        self.path = path

    def after_epoch(self):
        super().after_epoch()
        if self.new_best:
            print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
            torch.save(self.learn.model.state_dict(), os.path.join(self.path, self.fname + '.statedict'))


class SaveStatedictEveryNEpochCallback(Callback):
    """
    Callback that saves the models statedict at each Nth epoch.
    """
    _only_train_loop = True
    order = TrackerCallback.order + 1

    def __init__(self, n_to_save=1, path='', fname='model'):
        """
        Creates a SaveStatedictEveryNEpochCallback object.

        Output path is::
            os.path.join(path, f'{fname}-{epoch}.statedict')

        @arg n_to_save:
            When to save
        @arg path:
            Folder to save the statedict in
        @arg fname:
            Filename to use
        """
        super().__init__()
        self.fname = fname
        self.path = path
        self.n_to_save = n_to_save

    def after_epoch(self):
        if self.epoch % self.n_to_save == 0:
            print(f'Saving at epoch {self.epoch}')
            torch.save(self.learn.model.state_dict(), os.path.join(self.path, f'{self.fname}-{self.epoch}.statedict'))
