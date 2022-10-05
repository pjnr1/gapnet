import torch
import torch.nn as nn

from dnn_modelling.modules import HannAvgPool2d


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SaddlerEtAl2020(nn.Sequential):
    def __init__(self, conv_layers, fc_1_size, fc_out_size, input_shape=None, affine_bn=True):
        if input_shape is None:
            input_shape = [1, 100, 1000]
        super(SaddlerEtAl2020, self).__init__()
        self.conv_layers = nn.Sequential()

        conv_in_channels = input_shape[0]
        for i, cl in enumerate(conv_layers):
            self.add_module(f'conv_{i + 1}', nn.Conv2d(in_channels=conv_in_channels,
                                                       out_channels=cl['channels'],
                                                       kernel_size=cl['kernel_size']))
            self.add_module(f'relu_{i + 1}', nn.ReLU())
            self.add_module(f'pool_{i + 1}',
                            HannAvgPool2d(channels=cl['channels'],
                                          stride=cl['pool_stride']))
            self.add_module(f'norm_{i + 1}',
                            nn.BatchNorm2d(num_features=cl['channels'],
                                           affine=affine_bn))
            conv_in_channels = cl['channels']
        self.add_module('flatten', nn.Flatten())

        # Get number of input features for 'fc_1'
        with torch.no_grad():
            x = torch.zeros((1, input_shape[0], input_shape[1], input_shape[2]))
            for module in self:
                x = module(x)
            fc_1_in_features = x.shape[1]
            del x

        self.add_module('fc_1',
                        nn.Linear(in_features=fc_1_in_features,
                                  out_features=fc_1_size))
        self.add_module('relu_fc_1',
                        nn.ReLU())
        self.add_module('norm_fc_1',
                        nn.BatchNorm1d(num_features=fc_1_size,
                                       affine=affine_bn))
        self.add_module('dropout',
                        nn.Dropout(p=0.5))
        self.add_module('fc_out',
                        nn.Linear(in_features=fc_1_size,
                                  out_features=fc_out_size))
        self.add_module('softmax',
                        nn.Softmax(dim=-1))


def get_model(output_classes=700, arch='0191', input_shape=None):
    """
    The best performing architecture in the paper
    Also acts as an example on how to define the hyper-params in a json-ish style
    """
    archs = {'0191': {'conv_layers': [{'channels': 32,
                                       'kernel_size': [2, 83],
                                       'pool_stride': [1, 2]},
                                      {'channels': 64,
                                       'kernel_size': [1, 164],
                                       'pool_stride': [3, 7]},
                                      {'channels': 128,
                                       'kernel_size': [5, 9],
                                       'pool_stride': [1, 7]},
                                      {'channels': 256,
                                       'kernel_size': [4, 3],
                                       'pool_stride': [2, 1]},
                                      {'channels': 512,
                                       'kernel_size': [5, 2],
                                       'pool_stride': [1, 1]}],
                      'fc_1_size': 256},
             '0302': {'conv_layers': [{'channels': 32,
                                       'kernel_size': [1, 250],
                                       'pool_stride': [1, 5]},
                                      {'channels': 64,
                                       'kernel_size': [19, 11],
                                       'pool_stride': [1, 7]},
                                      {'channels': 128,
                                       'kernel_size': [12, 9],
                                       'pool_stride': [3, 1]},
                                      {'channels': 256,
                                       'kernel_size': [7, 7],
                                       'pool_stride': [1, 1]},
                                      {'channels': 512,
                                       'kernel_size': [5, 3],
                                       'pool_stride': [3, 1]}],
                      'fc_1_size': 1024},
             '0286': {'conv_layers': [{'channels': 64,
                                       'kernel_size': [1, 180],
                                       'pool_stride': [2, 6]},
                                      {'channels': 128,
                                       'kernel_size': [2, 37],
                                       'pool_stride': [1, 1]},
                                      {'channels': 128,
                                       'kernel_size': [15, 10],
                                       'pool_stride': [1, 1]}],
                      'fc_1_size': 512},
             'jl_gdt_1': {'conv_layers': [{'channels': 32,
                                           'kernel_size': [1, 128],
                                           'pool_stride': [1, 2]},
                                          {'channels': 64,
                                           'kernel_size': [19, 11],
                                           'pool_stride': [1, 7]},
                                          {'channels': 128,
                                           'kernel_size': [12, 9],
                                           'pool_stride': [3, 1]},
                                          {'channels': 256,
                                           'kernel_size': [7, 7],
                                           'pool_stride': [1, 1]},
                                          {'channels': 512,
                                           'kernel_size': [5, 3],
                                           'pool_stride': [3, 1]}],
                          'fc_1_size': 1024},
             'jl_gdt_2': {'conv_layers': [{'channels': 32,
                                           'kernel_size': [2, 64],
                                           'pool_stride': [1, 2]},
                                          {'channels': 64,
                                           'kernel_size': [1, 128],
                                           'pool_stride': [1, 4]},
                                          {'channels': 128,
                                           'kernel_size': [5, 9],
                                           'pool_stride': [1, 2]},
                                          {'channels': 256,
                                           'kernel_size': [4, 3],
                                           'pool_stride': [2, 1]},
                                          {'channels': 512,
                                           'kernel_size': [5, 2],
                                           'pool_stride': [1, 1]}],
                          'fc_1_size': 1024}}

    return SaddlerEtAl2020(**archs[arch], input_shape=input_shape, fc_out_size=output_classes)
