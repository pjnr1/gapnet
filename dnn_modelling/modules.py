import torch
import torch.nn as nn

from dnn_modelling.helpers import get_2d_zeropadding


class NonTorchModel:
    def __call__(self, x):
        self.forward(x)

    def forward(self, *args):
        raise NotImplementedError


class WeightedAvgPool2d(nn.Module):
    def __init__(self, channels, weights, stride=(1, 1), padding=(0, 0)):
        super(WeightedAvgPool2d, self).__init__()
        kernel_size = weights.shape
        self.conv = nn.Conv2d(in_channels=channels,
                              out_channels=channels,
                              groups=channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.conv.requires_grad_(False)
        self.shape_and_add_weights(weights)

    def forward(self, x):
        return self.conv(x)

    def shape_and_add_weights(self, w):
        conv_shape = self.conv.weight.data.shape
        w = torch.reshape(w, (1, 1, w.shape[0], w.shape[1]))
        w = torch.repeat_interleave(w, repeats=conv_shape[0], dim=0)
        w = torch.repeat_interleave(w, repeats=conv_shape[1], dim=1)

        self.conv.weight.data = w


class WeightedAvgPool2dSamePadding(WeightedAvgPool2d):
    def __init__(self, *args, **kwargs):
        super(WeightedAvgPool2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(get_2d_zeropadding(self.kernel_size))

    def forward(self, x):
        return super(WeightedAvgPool2dSamePadding, self).forward(self.zero_pad_2d(x))


class HannAvgPool2d(WeightedAvgPool2d):
    """
    As specified by Saddler et al 2020
    The Hann window H size follows from the stride:

    """

    def __init__(self, channels, stride=(1, 1)):
        kernel_size = [x if x == 1 else 4 * x for x in stride]
        weights = torch.outer(torch.hann_window(kernel_size[0]), torch.hann_window(kernel_size[1]))
        super(HannAvgPool2d, self).__init__(channels=channels,
                                            weights=weights,
                                            stride=stride,
                                            padding=(0, 0))
        self.zero_pad = nn.ZeroPad2d(get_2d_zeropadding(self.kernel_size))

    def forward(self, x):
        return self.conv(self.zero_pad(x))


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(get_2d_zeropadding(self.kernel_size))

    def forward(self, x):
        return super(Conv2dSamePadding, self).forward(self.zero_pad_2d(x))


class AvgPool2dSamePadding(nn.AvgPool2d):
    def __init__(self, *args, **kwargs):
        super(AvgPool2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(get_2d_zeropadding(self.kernel_size))

    def forward(self, x):
        return super(AvgPool2dSamePadding, self).forward(self.zero_pad_2d(x))


class MaxPool2dSamePadding(nn.MaxPool2d):
    def __init__(self, *args, **kwargs):
        super(MaxPool2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(get_2d_zeropadding(self.kernel_size))

    def forward(self, x):
        return super(MaxPool2dSamePadding, self).forward(self.zero_pad_2d(x))