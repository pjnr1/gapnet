import torch
import torch.nn as nn

from dnn_modelling.modules import Conv2dSamePadding
from dnn_modelling.modules import AvgPool2dSamePadding
from dnn_modelling.modules import MaxPool2dSamePadding
from dnn_modelling.helpers import weight_initialisation


class Lindahl_model_1(nn.Sequential):
    """
    Similar to Kell et al., but using temporally wider filters
    """
    def __init__(self, input_shape=None, output_classes=2, channel_factor=2, hidden_layer_size=1024, **kwargs):
        if input_shape is None:
            input_shape = [1, 100, 1000]
        super().__init__()
        self.channel_factor = channel_factor
        self.hidden_layer_size = hidden_layer_size
        self.conv_layers = [{'channels': 96 // self.channel_factor,
                             'kernel_size': (9, 12),
                             'stride': [3, 3]},
                            {'channels': 256 // self.channel_factor,
                             'kernel_size': [5, 8],
                             'stride': [2, 2]},
                            {'channels': 512 // self.channel_factor,
                             'kernel_size': [3, 5],
                             'stride': [1, 1]},
                            {'channels': 1024 // self.channel_factor,
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

        # Weights initialization
        self.apply(weight_initialisation)


class Lindahl_model_2(nn.Sequential):
    """
    Similar to Kell et al., but using temporal convolutions only
    """
    def __init__(self,
                 input_shape=None,
                 output_classes=2,
                 channel_factor=2,
                 hidden_layer_size=1024,
                 output_activation=None,
                 **kwargs):
        if input_shape is None:
            input_shape = [1, 100, 1000]
        if output_activation is None:
            output_activation = nn.Softmax(dim=-1)
        super().__init__()
        self.channel_factor = channel_factor
        self.hidden_layer_size = hidden_layer_size
        self.conv_layers = [{'channels': 1 * self.channel_factor,
                             'kernel_size': (1, 9),
                             'stride': [1, 3]},
                            {'channels': 2 * self.channel_factor,
                             'kernel_size': [1, 5],
                             'stride': [1, 2]},
                            {'channels': 4 * self.channel_factor,
                             'kernel_size': [1, 3],
                             'stride': [1, 1]},
                            {'channels': 8 * self.channel_factor,
                             'kernel_size': [1, 3],
                             'stride': [1, 1]},
                            {'channels': 8 * self.channel_factor,
                             'kernel_size': [1, 3],
                             'stride': [1, 1]}]

        conv_in_channels = input_shape[0]
        for i, cl in enumerate(self.conv_layers):
            self.add_module(f'conv_{i + 1}', Conv2dSamePadding(in_channels=conv_in_channels,
                                                               out_channels=cl['channels'],
                                                               kernel_size=cl['kernel_size'],
                                                               stride=cl['stride']))
            self.add_module(f'relu_{i + 1}', nn.ReLU())
            self.add_module(f'norm_{i + 1}', nn.LocalResponseNorm(size=5, alpha=1e-3, beta=0.75))
            self.add_module(f'pool_{i + 1}', AvgPool2dSamePadding((1, 3), 2))
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
        self.add_module('softmax', output_activation)

        # Weights initialization
        self.apply(weight_initialisation)


class Lindahl_model_3(nn.Sequential):
    """
    Similar to Kell et al., but using temporal convolutions only
    """
    def __init__(self,
                 input_shape=None,
                 output_classes=2,
                 channel_factor=4,
                 hidden_layer_size=1024,
                 bias=True,
                 dropout=0.0,
                 verbose=True,
                 output_activation=None,
                 **kwargs):  # eat all additional arguments
        if input_shape is None:
            input_shape = [1, 100, 1000]
        if output_activation is None:
            output_activation = nn.Softmax(dim=-1)
        super().__init__()
        self.channel_factor = channel_factor
        self.hidden_layer_size = hidden_layer_size
        self.bias = bias
        self.dropout = dropout

        self.conv_layers = [{'channels': 1 * self.channel_factor,
                             'kernel_size': (1, 16),
                             'stride': [1, 3]},
                            {'channels': 2 * self.channel_factor,
                             'kernel_size': [1, 12],
                             'stride': [1, 2]},
                            {'channels': 4 * self.channel_factor,
                             'kernel_size': [1, 8],
                             'stride': [1, 1]},
                            {'channels': 8 * self.channel_factor,
                             'kernel_size': [1, 4],
                             'stride': [1, 1]},
                            {'channels': 8 * self.channel_factor,
                             'kernel_size': [1, 4],
                             'stride': [1, 1]}]

        conv_in_channels = input_shape[0]
        for i, cl in enumerate(self.conv_layers):
            self.add_module(f'conv_{i + 1}', Conv2dSamePadding(in_channels=conv_in_channels,
                                                               out_channels=cl['channels'],
                                                               kernel_size=cl['kernel_size'],
                                                               stride=cl['stride'],
                                                               bias=self.bias))
            self.add_module(f'conv_{i + 1}_dropout', nn.Dropout2d(p=self.dropout))
            self.add_module(f'relu_{i + 1}', nn.ReLU())
            self.add_module(f'norm_{i + 1}', nn.LocalResponseNorm(size=5, alpha=1e-3, beta=0.75))
            self.add_module(f'pool_{i + 1}', AvgPool2dSamePadding((1, 3), 2))
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
        self.add_module('fc_1_dropout', nn.Dropout(p=self.dropout))
        self.add_module('fc_1_relu', nn.ReLU())
        self.add_module('fc_out',
                        nn.Linear(in_features=self.hidden_layer_size,
                                  out_features=output_classes,
                                  bias=self.bias))
        self.add_module('softmax', output_activation)

        # Weights initialization
        self.apply(weight_initialisation)

        if verbose:
            print('Initialised model "Lindahl3":')
            print('  input_shape   ', input_shape)
            print('  channel_factor', channel_factor)
            print('  bias          ', bias)
            print('  dropout       ', dropout)


class Lindahl_model_3b(nn.Sequential):
    """
    Similar to Kell et al., but using temporal convolutions only
    """
    def __init__(self,
                 input_shape=None,
                 output_classes=2,
                 channel_factor=4,
                 hidden_layer_size=1024,
                 bias=True,
                 dropout=0.0,
                 verbose=True,
                 output_activation=None,
                 **kwargs):  # eat all additional arguments
        if input_shape is None:
            input_shape = [1, 100, 1000]
        if output_activation is None:
            output_activation = nn.Softmax(dim=-1)
        super().__init__()
        self.channel_factor = channel_factor
        self.hidden_layer_size = hidden_layer_size
        self.bias = bias
        self.dropout = dropout

        self.conv_layers = [{'channels': 1 * self.channel_factor,
                             'kernel_size': (1, 16),
                             'stride': [1, 1]},
                            {'channels': 2 * self.channel_factor,
                             'kernel_size': [1, 12],
                             'stride': [1, 1]},
                            {'channels': 4 * self.channel_factor,
                             'kernel_size': [1, 8],
                             'stride': [1, 1]},
                            {'channels': 8 * self.channel_factor,
                             'kernel_size': [1, 4],
                             'stride': [1, 1]},
                            {'channels': 8 * self.channel_factor,
                             'kernel_size': [1, 4],
                             'stride': [1, 1]}]

        conv_in_channels = input_shape[0]
        for i, cl in enumerate(self.conv_layers):
            self.add_module(f'conv_{i + 1}', Conv2dSamePadding(in_channels=conv_in_channels,
                                                               out_channels=cl['channels'],
                                                               kernel_size=cl['kernel_size'],
                                                               stride=cl['stride'],
                                                               bias=self.bias))
            self.add_module(f'conv_{i + 1}_dropout', nn.Dropout2d(p=self.dropout))
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
        self.add_module('fc_1_dropout', nn.Dropout(p=self.dropout))
        self.add_module('fc_1_relu', nn.ReLU())
        self.add_module('fc_out',
                        nn.Linear(in_features=self.hidden_layer_size,
                                  out_features=output_classes,
                                  bias=self.bias))
        self.add_module('softmax', output_activation)

        # Weights initialization
        self.apply(weight_initialisation)

        if verbose:
            print(f'Initialised model "{type(self).__name__}":')
            print('  input_shape   ', input_shape)
            print('  channel_factor', channel_factor)
            print('  bias          ', bias)
            print('  dropout       ', dropout)


class Lindahl_model_3c(nn.Sequential):
    """
    Similar to Kell et al., but using temporal convolutions only
    """
    def __init__(self,
                 input_shape=None,
                 output_classes=2,
                 channel_factor=4,
                 hidden_layer_size=1024,
                 bias=True,
                 dropout=0.0,
                 verbose=True,
                 output_activation=None,
                 **kwargs):  # eat all additional arguments
        if input_shape is None:
            input_shape = [1, 100, 1000]
        if output_activation is None:
            output_activation = nn.Softmax(dim=-1)
        super().__init__()
        self.channel_factor = channel_factor
        self.hidden_layer_size = hidden_layer_size
        self.bias = bias
        self.dropout = dropout

        self.conv_layers = [{'channels': 1 * self.channel_factor,
                             'kernel_size': (1, 16),
                             'stride': [1, 1]},
                            {'channels': 2 * self.channel_factor,
                             'kernel_size': [1, 12],
                             'stride': [1, 1]},
                            {'channels': 4 * self.channel_factor,
                             'kernel_size': [1, 8],
                             'stride': [1, 1]},
                            {'channels': 8 * self.channel_factor,
                             'kernel_size': [1, 4],
                             'stride': [1, 1]},
                            {'channels': 8 * self.channel_factor,
                             'kernel_size': [1, 4],
                             'stride': [1, 1]},
                            {'channels': 4 * self.channel_factor,
                             'kernel_size': [3, 3],
                             'stride': [1, 1]},
                            {'channels': 2 * self.channel_factor,
                             'kernel_size': [3, 3],
                             'stride': [1, 1]}]

        conv_in_channels = input_shape[0]
        for i, cl in enumerate(self.conv_layers):
            self.add_module(f'conv_{i + 1}', Conv2dSamePadding(in_channels=conv_in_channels,
                                                               out_channels=cl['channels'],
                                                               kernel_size=cl['kernel_size'],
                                                               stride=cl['stride'],
                                                               bias=self.bias))
            self.add_module(f'conv_{i + 1}_dropout', nn.Dropout2d(p=self.dropout))
            self.add_module(f'relu_{i + 1}', nn.ReLU())
            self.add_module(f'norm_{i + 1}', nn.LocalResponseNorm(size=5, alpha=1e-3, beta=0.75))
            self.add_module(f'pool_{i + 1}', MaxPool2dSamePadding((3, 3), 2))
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
        self.add_module('fc_1_dropout', nn.Dropout(p=self.dropout))
        self.add_module('fc_1_relu', nn.ReLU())
        self.add_module('fc_out',
                        nn.Linear(in_features=self.hidden_layer_size,
                                  out_features=output_classes,
                                  bias=self.bias))
        self.add_module('softmax', output_activation)

        # Weights initialization
        self.apply(weight_initialisation)

        if verbose:
            print('Initialised model "Lindahl3":')
            print('  input_shape   ', input_shape)
            print('  channel_factor', channel_factor)
            print('  bias          ', bias)
            print('  dropout       ', dropout)


class Lindahl_model_4(nn.Sequential):
    """
    Similar to Kell et al., but using temporal convolutions only
    """
    def __init__(self,
                 input_shape=None,
                 output_classes=2,
                 channel_factor=4,
                 hidden_layer_size=512,
                 bias=True,
                 dropout=0.0,
                 verbose=True,
                 output_activation=None,
                 **kwargs):  # eat all additional arguments
        if input_shape is None:
            input_shape = [1, 100, 1000]
        if output_activation is None:
            output_activation = nn.Softmax(dim=-1)
        super().__init__()
        self.channel_factor = channel_factor
        self.hidden_layer_size = hidden_layer_size
        self.bias = bias
        self.dropout = dropout

        self.conv_layers = [{'channels': 1 * self.channel_factor,
                             'kernel_size': (5, 5),
                             'stride': [2, 2]},
                            {'channels': 2 * self.channel_factor,
                             'kernel_size': [3, 3],
                             'stride': [1, 1]},
                            {'channels': 4 * self.channel_factor,
                             'kernel_size': [3, 3],
                             'stride': [1, 1]},
                            {'channels': 8 * self.channel_factor,
                             'kernel_size': [3, 3],
                             'stride': [1, 1]},
                            {'channels': 8 * self.channel_factor,
                             'kernel_size': [3, 3],
                             'stride': [1, 1]}]

        conv_in_channels = input_shape[0]
        for i, cl in enumerate(self.conv_layers):
            self.add_module(f'conv_{i + 1}', Conv2dSamePadding(in_channels=conv_in_channels,
                                                               out_channels=cl['channels'],
                                                               kernel_size=cl['kernel_size'],
                                                               stride=cl['stride'],
                                                               bias=self.bias))
            self.add_module(f'conv_{i + 1}_dropout', nn.Dropout2d(p=self.dropout))
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
        self.add_module('fc_1_dropout', nn.Dropout(p=self.dropout))
        self.add_module('fc_1_relu', nn.ReLU())
        self.add_module('fc_out',
                        nn.Linear(in_features=self.hidden_layer_size,
                                  out_features=output_classes,
                                  bias=self.bias))
        self.add_module('softmax', output_activation)

        # Weights initialization
        self.apply(weight_initialisation)

        if verbose:
            print(f'Initialised model "{type(self).__name__}":')
            print('  input_shape   ', input_shape)
            print('  channel_factor', channel_factor)
            print('  bias          ', bias)
            print('  dropout       ', dropout)


class Lindahl_model_5(nn.Sequential):
    """
    Similar to Kell et al., but using temporal convolutions only
    """
    def __init__(self,
                 input_shape=None,
                 output_classes=2,
                 channel_factor=4,
                 hidden_layer_size=512,
                 bias=True,
                 dropout=0.0,
                 verbose=True,
                 output_activation=None,
                 **kwargs):  # eat all additional arguments
        if input_shape is None:
            input_shape = [1, 100, 1000]
        if output_activation is None:
            output_activation = nn.Softmax(dim=-1)
        super().__init__()
        self.channel_factor = channel_factor
        self.hidden_layer_size = hidden_layer_size
        self.bias = bias
        self.dropout = dropout

        self.conv_layers = [{'channels': 1 * self.channel_factor,
                             'kernel_size': (1, 5),
                             'stride': [1, 2]},
                            {'channels': 2 * self.channel_factor,
                             'kernel_size': [1, 3],
                             'stride': [1, 1]},
                            {'channels': 4 * self.channel_factor,
                             'kernel_size': [1, 3],
                             'stride': [1, 1]},
                            {'channels': 8 * self.channel_factor,
                             'kernel_size': [1, 3],
                             'stride': [1, 1]},
                            {'channels': 8 * self.channel_factor,
                             'kernel_size': [1, 3],
                             'stride': [1, 1]}]

        conv_in_channels = input_shape[0]
        for i, cl in enumerate(self.conv_layers):
            self.add_module(f'conv_{i + 1}', Conv2dSamePadding(in_channels=conv_in_channels,
                                                               out_channels=cl['channels'],
                                                               kernel_size=cl['kernel_size'],
                                                               stride=cl['stride'],
                                                               bias=self.bias))
            self.add_module(f'conv_{i + 1}_dropout', nn.Dropout2d(p=self.dropout))
            self.add_module(f'relu_{i + 1}', nn.ReLU())
            self.add_module(f'norm_{i + 1}', nn.LocalResponseNorm(size=5, alpha=1e-3, beta=0.75))
            self.add_module(f'pool_{i + 1}', AvgPool2dSamePadding((1, 3), 2))
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
        self.add_module('fc_1_dropout', nn.Dropout(p=self.dropout))
        self.add_module('fc_1_relu', nn.ReLU())
        self.add_module('fc_out',
                        nn.Linear(in_features=self.hidden_layer_size,
                                  out_features=output_classes,
                                  bias=self.bias))
        self.add_module('softmax', output_activation)

        # Weights initialization
        self.apply(weight_initialisation)

        if verbose:
            print(f'Initialised model "{type(self).__name__}":')
            print('  input_shape   ', input_shape)
            print('  channel_factor', channel_factor)
            print('  bias          ', bias)
            print('  dropout       ', dropout)


class Lindahl_model_6(nn.Sequential):
    """
    Similar to Kell et al., but using temporal convolutions only
    """
    def __init__(self,
                 input_shape=None,
                 output_classes=2,
                 channel_factor=4,
                 hidden_layer_size=1024,
                 bias=True,
                 dropout=0.0,
                 verbose=True,
                 output_activation=None,
                 **kwargs):  # eat all additional arguments
        if input_shape is None:
            input_shape = [1, 100, 1000]
        if output_activation is None:
            output_activation = nn.Softmax(dim=-1)
        super().__init__()
        self.channel_factor = channel_factor
        self.hidden_layer_size = hidden_layer_size
        self.bias = bias
        self.dropout = dropout

        self.conv_layers = [{'channels': 1 * self.channel_factor,
                             'kernel_size': (5, 16),
                             'stride': [2, 3]},
                            {'channels': 2 * self.channel_factor,
                             'kernel_size': [3, 12],
                             'stride': [1, 2]},
                            {'channels': 4 * self.channel_factor,
                             'kernel_size': [3, 8],
                             'stride': [1, 1]},
                            {'channels': 8 * self.channel_factor,
                             'kernel_size': [3, 4],
                             'stride': [1, 1]},
                            {'channels': 8 * self.channel_factor,
                             'kernel_size': [3, 4],
                             'stride': [1, 1]}]

        conv_in_channels = input_shape[0]
        for i, cl in enumerate(self.conv_layers):
            self.add_module(f'conv_{i + 1}', Conv2dSamePadding(in_channels=conv_in_channels,
                                                               out_channels=cl['channels'],
                                                               kernel_size=cl['kernel_size'],
                                                               stride=cl['stride'],
                                                               bias=self.bias))
            self.add_module(f'conv_{i + 1}_dropout', nn.Dropout2d(p=self.dropout))
            self.add_module(f'relu_{i + 1}', nn.ReLU())
            self.add_module(f'norm_{i + 1}', nn.LocalResponseNorm(size=5, alpha=1e-3, beta=0.75))
            self.add_module(f'pool_{i + 1}', AvgPool2dSamePadding((1, 3), 2))
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
        self.add_module('fc_1_dropout', nn.Dropout(p=self.dropout))
        self.add_module('fc_1_relu', nn.ReLU())
        self.add_module('fc_out',
                        nn.Linear(in_features=self.hidden_layer_size,
                                  out_features=output_classes,
                                  bias=self.bias))
        self.add_module('softmax', output_activation)

        # Weights initialization
        self.apply(weight_initialisation)

        if verbose:
            print(f'Initialised model "{type(self).__name__}":')
            print('  input_shape   ', input_shape)
            print('  channel_factor', channel_factor)
            print('  bias          ', bias)
            print('  dropout       ', dropout)

