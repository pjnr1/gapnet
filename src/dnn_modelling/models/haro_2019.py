import torch
import torch.nn as nn

from dnn_modelling.modules import Conv2dSamePadding, AvgPool2dSamePadding


class HaroEtAl2020_noSplit_nClass(nn.Sequential):
    def __init__(self, input_shape=None, output_classes=2):
        if input_shape is None:
            input_shape = [1, 100, 1000]
        super().__init__()
        self.conv_layers = [{'channels': 96,
                             'kernel_size': [5, 5],
                             'stride': [1, 1]},
                            {'channels': 256,
                             'kernel_size': [3, 3],
                             'stride': [1, 1]},
                            {'channels': 512,
                             'kernel_size': [3, 3],
                             'stride': [1, 1]},
                            {'channels': 1024,
                             'kernel_size': [3, 3],
                             'stride': [1, 1]},
                            {'channels': 512,
                             'kernel_size': [3, 3],
                             'stride': [1, 1]}]

        conv_in_channels = input_shape[0]
        for i, cl in enumerate(self.conv_layers):
            self.add_module(f'conv_{i + 1}', Conv2dSamePadding(in_channels=conv_in_channels,
                                                               out_channels=cl['channels'],
                                                               kernel_size=cl['kernel_size'],
                                                               stride=cl['stride']))
            self.add_module(f'relu_{i + 1}', nn.ReLU())
            self.add_module(f'norm_{i + 1}', nn.LocalResponseNorm(size=5, alpha=1e-3, beta=0.75))
            if i < 2:
                self.add_module(f'pool_{i + 1}', AvgPool2dSamePadding((3, 3), 2))
            conv_in_channels = cl['channels']
        self.add_module('flatten', nn.Flatten())

        # Get number of input features for 'fc_1'
        with torch.no_grad():
            x = torch.zeros((1, input_shape[0], input_shape[1], input_shape[2]))
            for module in self:
                x = module(x)
            fc_1_in_features = x.shape[1]
            del x

        # fc
        self.add_module('fc_1', nn.Linear(in_features=fc_1_in_features,
                                          out_features=1024))
        self.add_module('fc_1_relu', nn.ReLU())
        self.add_module('fc_out',
                        nn.Linear(in_features=1024,
                                  out_features=output_classes))
        self.add_module('softmax',
                        nn.Softmax(dim=-1))


def get_model(output_classes=2, input_shape=None):
    return HaroEtAl2020_noSplit_nClass(output_classes=output_classes,
                                       input_shape=input_shape)
