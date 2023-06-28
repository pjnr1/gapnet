import torch
import torch.nn as nn
import torch.nn.functional as F

methods = {'max': lambda x: x.amax(dim=-1),
           'min': lambda x: x.amin(dim=-1),
           'max-min': lambda x: x.amax(dim=-1) - x.amin(dim=-1)}

windows = {
    'hann': lambda x: torch.reshape(torch.hann_window(x), (1, 1, x)),
    'ones': lambda x: torch.ones(1, 1, x),
    'rect': lambda x: torch.ones(1, 1, x),
}


class NeuralMetric(nn.Module):
    def __init__(self,
                 window_size=1,
                 method=None,
                 window='rect',
                 output_classes=2,
                 diff=True,
                 **kwargs):
        super().__init__()
        if method is None:
            method = 'max'
        if method not in methods.keys():
            raise ValueError(f'method is not valid ({method}). Valid methods are: {methods.keys()}')
        if not isinstance(window_size, int):
            window_size = int(window_size)
        self.method = methods[method]
        self.conv_weight = windows[window](window_size)
        self.conv_weight /= torch.sum(self.conv_weight)
        self.diff = diff
        self.output_classes = output_classes

    def forward(self, x):
        x = x.mean(-2)  # across CF
        x = F.conv1d(x, weight=self.conv_weight, padding='same')  # window in temporal domain
        if self.diff:
            x = torch.diff(x, dim=-1)
        x = x[:, :, 300:600]
        x = self.method(x)
        x = x.expand(-1, self.output_classes)
        return x
