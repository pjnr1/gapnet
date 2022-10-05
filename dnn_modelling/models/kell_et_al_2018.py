import torch
import torch.nn as nn

from dnn_modelling.modules import Conv2dSamePadding, AvgPool2dSamePadding

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class KellEtAl2018(nn.Module):
    def __init__(self):
        super(KellEtAl2018, self).__init__()

        # Shared stack
        self.shared_stack = nn.Sequential(
            # Conv1
            Conv2dSamePadding(in_channels=1,
                              out_channels=96,
                              kernel_size=(9, 9),
                              stride=(3, 3)),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-3, beta=0.75),
            AvgPool2dSamePadding((3, 3), 2),
            # Conv2
            Conv2dSamePadding(in_channels=96,
                              out_channels=256,
                              kernel_size=(5, 5),
                              stride=(2, 2)),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-3, beta=0.75),
            AvgPool2dSamePadding((3, 3), 2),
            # Conv3
            Conv2dSamePadding(in_channels=256,
                              out_channels=512,
                              kernel_size=(3, 3),
                              stride=(1, 1)),
            nn.ReLU(),
        )

        def split_stack(out_channels):
            return nn.Sequential(
                # conv4
                Conv2dSamePadding(in_channels=512,
                                  out_channels=1024,
                                  kernel_size=(3, 3),
                                  stride=(1, 1)),
                nn.ReLU(),
                # conv5
                Conv2dSamePadding(in_channels=1024,
                                  out_channels=512,
                                  kernel_size=(3, 3),
                                  stride=(1, 1)),
                nn.ReLU(),
                # pool3
                AvgPool2dSamePadding((3, 3), 2),
                # fc
                nn.Flatten(),
                nn.Linear(in_features=6 * 6 * 512,
                          out_features=1024),
                nn.ReLU(),
                nn.Linear(in_features=1024,
                          out_features=out_channels),
                nn.Softmax(dim=-1)
            )

        self.W_stack = split_stack(589)
        self.G_stack = split_stack(43)

    def forward(self, x):
        x = self.shared_stack(x)
        x_W = self.W_stack(x)
        x_G = self.G_stack(x)
        return x_W, x_G


class KellEtAl2018_noSplit_nClass(nn.Sequential):
    def __init__(self, input_shape=None, output_classes=2):
        if input_shape is None:
            input_shape = [1, 100, 1000]
        super(KellEtAl2018_noSplit_nClass, self).__init__()
        self.conv_layers = [{'channels': 96,
                             'kernel_size': [9, 9],
                             'stride': [3, 3]},
                            {'channels': 256,
                             'kernel_size': [5, 5],
                             'stride': [2, 2]},
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


class KellEtAl2018_smaller(nn.Sequential):
    """
    Same as KellEtAll2018 no split, n-class but with a reduced number of channels and possibility to set size of hidden
    layer
    """
    def __init__(self, input_shape=None, output_classes=2, channel_factor=2, hidden_layer_size=1024):
        if input_shape is None:
            input_shape = [1, 100, 1000]
        super().__init__()
        self.channel_factor = channel_factor
        self.hidden_layer_size = hidden_layer_size
        self.conv_layers = [{'channels': 96 // self.channel_factor,
                             'kernel_size': [9, 9],
                             'stride': [3, 3]},
                            {'channels': 256 // self.channel_factor,
                             'kernel_size': [5, 5],
                             'stride': [2, 2]},
                            {'channels': 512 // self.channel_factor,
                             'kernel_size': [3, 3],
                             'stride': [1, 1]},
                            {'channels': 1024 // self.channel_factor,
                             'kernel_size': [3, 3],
                             'stride': [1, 1]},
                            {'channels': 512 // self.channel_factor,
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
                                          out_features=self.hidden_layer_size))
        self.add_module('fc_1_relu', nn.ReLU())
        self.add_module('fc_out',
                        nn.Linear(in_features=self.hidden_layer_size,
                                  out_features=output_classes))
        self.add_module('softmax',
                        nn.Softmax(dim=-1))

