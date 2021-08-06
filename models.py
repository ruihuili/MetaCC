#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l

from scipy.stats import truncnorm


def truncated_normal_(tensor, mean=0.0, std=1.0):
    # PT doesn't have truncated normal.
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/18
    values = truncnorm.rvs(-2, 2, size=tensor.shape)
    values = mean + std * values
    tensor.copy_(torch.from_numpy(values))
    return tensor


def fc_init_(module):
    if hasattr(module, 'weight') and module.weight is not None:
        truncated_normal_(module.weight.data, mean=0.0, std=0.01)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.constant_(module.bias.data, 0.0)
    return module


def maml_init_(module):
    torch.nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    torch.nn.init.constant_(module.bias.data, 0.0)
    return module


class LinearBlock(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearBlock, self).__init__()
        self.relu = torch.nn.ReLU()
        self.normalize = torch.nn.BatchNorm1d(
            output_size,
            affine=True,
            momentum=0.999,
            eps=1e-3,
            track_running_stats=False,
        )
        self.linear = torch.nn.Linear(input_size, output_size)
        fc_init_(self.linear)

    def forward(self, x):
        x = self.linear(x)
        x = self.normalize(x)
        x = self.relu(x)
        return x


class ConvBlock(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 max_pool=True,
                 max_pool_factor=1.0):
        super(ConvBlock, self).__init__()
        stride = (1, int(2 * max_pool_factor))
        if max_pool:
            self.max_pool = torch.nn.MaxPool2d(
                kernel_size=stride,
                stride=stride,
                ceil_mode=False,
            )
            stride = (1, 1)
        else:
            self.max_pool = lambda x: x
        self.normalize = torch.nn.BatchNorm2d(
            out_channels,
            affine=True,
            # eps=1e-3,
            # momentum=0.999,
            # track_running_stats=False,
        )
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = torch.nn.ReLU()

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=1,
            bias=True,
        )
        maml_init_(self.conv)

    def forward(self, x):
        # print('in block',x.shape)
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        # print('out block',x.shape)
        return x


class ConvBase(torch.nn.Sequential):

    # NOTE:
    #     Omniglot: hidden=64, channels=1, no max_pool
    #     MiniImagenet: hidden=32, channels=3, max_pool

    def __init__(self,
                 output_size,
                 hidden=64,
                 channels=1,
                 max_pool=False,
                 layers=4,
                 max_pool_factor=1.0):
        core = [ConvBlock(channels,
                          hidden,
                          (3, 3),
                          max_pool=max_pool,
                          max_pool_factor=max_pool_factor),
                ]
        for _ in range(layers - 1):
            core.append(ConvBlock(hidden,
                                  hidden,
                                  kernel_size=(3, 3),
                                  max_pool=max_pool,
                                  max_pool_factor=max_pool_factor))
        super(ConvBase, self).__init__(*core)


class ConvBase2(torch.nn.Sequential):

    # NOTE:
    #     Omniglot: hidden=64, channels=1, no max_pool
    #     MiniImagenet: hidden=32, channels=3, max_pool

    def __init__(self,
                 output_size,
                 hidden=64,
                 channels=1,
                 max_pool=False,
                 layers=4,
                 max_pool_factor=1.0):
        core = []
        for _ in range(layers - 1):
            core.append(ConvBlock(hidden,
                                  hidden,
                                  kernel_size=(3, 3),
                                  max_pool=max_pool,
                                  max_pool_factor=max_pool_factor))
        super(ConvBase2, self).__init__(*core)


class CNN4(torch.nn.Module):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/cnn4.py)

    **Description**

    The convolutional network commonly used for MiniImagenet, as described by Ravi et Larochelle, 2017.

    **References**

    1. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=64) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.

    **Example**
    ~~~python
    model = CNN4(output_size=20, hidden_size=128, layers=3)
    ~~~
    """

    def __init__(self, output_size, hidden_size=64, layers=4, features_only=False):
        super(CNN4, self).__init__()
        base = ConvBase(
            output_size=hidden_size,
            hidden=hidden_size,
            channels=1,
            max_pool=False,
            layers=layers,
            max_pool_factor=4 // layers,
        )
        self.features = torch.nn.Sequential(
            base,
#             l2l.nn.Flatten(),
        )
        self.classifier = torch.nn.Linear(
            hidden_size,
            1,
            bias=True,
        )
        maml_init_(self.classifier)
        self.hidden_size = hidden_size
        self.features_only = features_only
        

    def forward(self, x):
        x = self.features(x)
        x = torch.squeeze(x).permute(0, 2, 1)
        # print(x.shape)
        if self.features_only:
            return x
        
        unflatten = torch.nn.Unflatten(0, (x.shape[0:2]))
        # print(x.shape)
        x = torch.flatten(x, 0, 1)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        x = unflatten(x).squeeze()
        # print(x.shape)
        return x


class CNN4_2(torch.nn.Module):
    def __init__(self, output_size, hidden_size=64, layers=4, features_only=False):
        super(CNN4_2, self).__init__()
        self.first_layer = ConvBlock(1,
                                     hidden_size,
                                     (3, 3),
                                     max_pool=False,
                                     max_pool_factor=4 // layers)
        base = ConvBase2(
            output_size=hidden_size,
            hidden=hidden_size,
            channels=1,
            max_pool=False,
            layers=layers,
            max_pool_factor=4 // layers,
        )
        self.features = torch.nn.Sequential(
            base,
            l2l.nn.Flatten(),
        )
        self.classifier = torch.nn.Linear(
            hidden_size,
            output_size,
            bias=True,
        )
        maml_init_(self.classifier)
        self.hidden_size = hidden_size
        self.features_only = features_only

    def forward(self, x):
        x = self.first_layer(x)
        x = self.features(x)
        if self.features_only:
            return x
        x = self.classifier(x)
        return x

class CNN4_layer_by_layer(torch.nn.Module):
    def __init__(self, output_size, hidden_size=64, layers=4, features_only=False):
        super(CNN4_layer_by_layer, self).__init__()
        self.layer_1 = ConvBlock(1,
                                     hidden_size,
                                     (3, 3),
                                     max_pool=False,
                                     max_pool_factor=4 // layers)
        self.layer_2 = ConvBlock(hidden_size,
                                     hidden_size,
                                     (3, 3),
                                     max_pool=False,
                                     max_pool_factor=4 // layers)
        self.layer_3 = ConvBlock(hidden_size,
                                     hidden_size,
                                     (3, 3),
                                     max_pool=False,
                                     max_pool_factor=4 // layers)
        self.layer_4 = ConvBlock(hidden_size,
                                     hidden_size,
                                     (3, 3),
                                     max_pool=False,
                                     max_pool_factor=4 // layers)

        self.layer_4 = torch.nn.Sequential(self.layer_4, l2l.nn.Flatten())

        
        self.classifier = torch.nn.Linear(
            hidden_size,
            output_size,
            bias=True,
        )
        maml_init_(self.classifier)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.classifier(x)
        return x


class CNN4Encoder(torch.nn.Module):
    def __init__(self, hidden_size=64, layers=4, features_only=False):
        super(CNN4Encoder, self).__init__()
        base = ConvBase(
            output_size=hidden_size,
            hidden=hidden_size,
            channels=1,
            max_pool=False,
            layers=layers,
            max_pool_factor=4 // layers,
        )
        self.features = torch.nn.Sequential(
            base,
            l2l.nn.Flatten(),
        )

    def forward(self, x):
        x = self.features(x)
        return x


# model used by CAVIA
class CondConvNet(nn.Module):
    def __init__(self,
                 num_classes,
                 num_context_params,
                 context_in,
                 num_film_hidden_layers,    
                 initialisation,
                 device,
                 channel = 1,
                 hidden_size=64,
                 max_pool = False
                 ):
        super(CondConvNet, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.max_pool = max_pool
        self.num_context_params = num_context_params
        self.context_in = context_in
        self.num_film_hidden_layers = num_film_hidden_layers
        self.kernel_size = 3

        # -- shared network --
        max_pool_factor= 1.0
        stride = (int(2 * max_pool_factor), int(2 * max_pool_factor))
        # stride = 1
        padding = 1
        self.channel = channel

        # conv-layers
        self.conv1 = nn.Conv2d(self.channel, self.hidden_size, self.kernel_size, stride=stride,
                               padding=padding).to(device)
        self.conv2 = nn.Conv2d(self.hidden_size, self.hidden_size, self.kernel_size, stride=stride, padding=padding).to(
            device)
        self.conv3 = nn.Conv2d(self.hidden_size, self.hidden_size, self.kernel_size, stride=stride, padding=padding).to(
            device)
        if not self.max_pool:
            self.conv4 = nn.Conv2d(self.hidden_size, self.hidden_size, self.kernel_size, stride=stride, 
                                padding=padding).to(device)
        else:
            self.conv4 = nn.Conv2d(self.hidden_size, self.hidden_size, self.kernel_size, stride=stride,
                                   padding=padding).to(device)

        # # batch norm
        self.bn1 = nn.BatchNorm2d(self.hidden_size, track_running_stats=False).to(device)
        self.bn2 = nn.BatchNorm2d(self.hidden_size, track_running_stats=False).to(device)
        self.bn3 = nn.BatchNorm2d(self.hidden_size, track_running_stats=False).to(device)
        self.bn4 = nn.BatchNorm2d(self.hidden_size, track_running_stats=False).to(device)

        # initialise weights for the fully connected layer

        # self.fc1 = torch.nn.Linear(hidden_size,output_size, bias=True)

        # if imsize == 84:
        #     self.fc1 = nn.Linear(5 * 5 * self.hidden_size + int(context_in[4]) * num_context_params, self.num_classes).to(device)
        # elif imsize == 28:
        #     self.fc1 = nn.Linear(self.hidden_size + int(context_in[4]) * num_context_params, self.num_classes).to(device)
        # elif imsize == 10:
        self.fc1 = nn.Linear(self.hidden_size + int(context_in[4]) * num_context_params, self.num_classes).to(device)
        # else:
        #     raise NotImplementedError('Cannot handle image size.')

        # -- additions to enable context parameters at convolutional layers --

        # for each layer where we have context parameters, initialise a FiLM layer
        if self.context_in[0]:
            self.film1 = nn.Linear(self.num_context_params, self.hidden_size * 2).to(device)
            if self.num_film_hidden_layers == 1:
                self.film11 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2).to(device)
        if self.context_in[1]:
            self.film2 = nn.Linear(self.num_context_params, self.hidden_size * 2).to(device)
            if self.num_film_hidden_layers == 1:
                self.film22 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2).to(device)
        if self.context_in[2]:
            self.film3 = nn.Linear(self.num_context_params, self.hidden_size * 2).to(device)
            if self.num_film_hidden_layers == 1:
                self.film33 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2).to(device)
        if self.context_in[3]:
            self.film4 = nn.Linear(self.num_context_params, self.hidden_size * 2).to(device)
            if self.num_film_hidden_layers == 1:
                self.film44 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2).to(device)

        # parameter initialisation (if different than standard pytorch one)
        if initialisation != 'standard':
            self.init_params(initialisation)

        # initialise context parameters
        self.context_params = torch.zeros(size=[self.num_context_params], requires_grad=True).to(device)

    def init_params(self, initialisation):

        # convolutional weights

        if initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu', self.conv1.weight))
            torch.nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu', self.conv2.weight))
            torch.nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('relu', self.conv3.weight))
            torch.nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('relu', self.conv4.weight))
        elif initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.conv4.weight, nonlinearity='relu')

        # convolutional bias

        self.conv1.bias.data.fill_(0)
        self.conv2.bias.data.fill_(0)
        self.conv3.bias.data.fill_(0)
        self.conv4.bias.data.fill_(0)

        # fully connected weights at the end

        if initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('linear', self.fc1.weight))
        elif initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='linear')

        # fully connected bias

        self.fc1.bias.data.fill_(0)

        # FiLM layer weights

        if self.context_in[0] and initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.film1.weight, gain=nn.init.calculate_gain('linear', self.film1.weight))
        elif self.context_in[0] and initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.film1.weight, nonlinearity='linear')

        if self.context_in[1] and initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.film2.weight, gain=nn.init.calculate_gain('linear', self.film2.weight))
        elif self.context_in[1] and initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.film2.weight, nonlinearity='linear')

        if self.context_in[2] and initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.film3.weight, gain=nn.init.calculate_gain('linear', self.film3.weight))
        elif self.context_in[2] and initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.film3.weight, nonlinearity='linear')

        if self.context_in[3] and initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.film4.weight, gain=nn.init.calculate_gain('linear', self.film4.weight))
        elif self.context_in[3] and initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.film4.weight, nonlinearity='linear')

    def reset_context_params(self):
        self.context_params = self.context_params.detach() * 0
        self.context_params.requires_grad = True

    def forward(self, x):

        # pass through convolutional layer
        h1 = self.conv1(x)
        # batchnorm
        h1 = self.bn1(h1)
        # do max-pooling (for imagenet)
        if self.max_pool:
            h1 = F.max_pool2d(h1, kernel_size=2)
        # if we have context parameters, adjust conv output using FiLM variables
        if self.context_in[0]:
            # FiLM it: forward through film layer to get scale and shift parameter
            film1 = self.film1(self.context_params)
            if self.num_film_hidden_layers == 1:
                film1 = self.film11(F.relu(film1))
            gamma1 = film1[:self.hidden_size].view(1, -1, 1, 1)
            beta1 = film1[self.hidden_size:].view(1, -1, 1, 1)
            # transform feature map
            h1 = gamma1 * h1 + beta1
        # pass through ReLu activation function
        h1 = F.relu(h1)

        h2 = self.conv2(h1)
        h2 = self.bn2(h2)
        if self.max_pool:
            h2 = F.max_pool2d(h2, kernel_size=2)
        if self.context_in[1]:
            film2 = self.film2(self.context_params)
            if self.num_film_hidden_layers == 1:
                film2 = self.film22(F.relu(film2))
            gamma2 = film2[:self.hidden_size].view(1, -1, 1, 1)
            beta2 = film2[self.hidden_size:].view(1, -1, 1, 1)
            h2 = gamma2 * h2 + beta2
        h2 = F.relu(h2)

        h3 = self.conv3(h2)
        h3 = self.bn3(h3)
        if self.max_pool:
            h3 = F.max_pool2d(h3, kernel_size=2)
        if self.context_in[2]:
            film3 = self.film3(self.context_params)
            if self.num_film_hidden_layers == 1:
                film3 = self.film33(F.relu(film3))
            gamma3 = film3[:self.hidden_size].view(1, -1, 1, 1)
            beta3 = film3[self.hidden_size:].view(1, -1, 1, 1)
            h3 = gamma3 * h3 + beta3
        h3 = F.relu(h3)

        h4 = self.conv4(h3)
        h4 = self.bn4(h4)
        if self.max_pool:
            h4 = F.max_pool2d(h4, kernel_size=2)
        if self.context_in[3]:
            film4 = self.film4(self.context_params)
            if self.num_film_hidden_layers == 1:
                film4 = self.film44(F.relu(film4))
            gamma4 = film4[:self.hidden_size].view(1, -1, 1, 1)
            beta4 = film4[self.hidden_size:].view(1, -1, 1, 1)
            h4 = gamma4 * h4 + beta4
        h4 = F.relu(h4)

        # flatten
        h4 = h4.view(h4.size(0), -1)

        if self.context_in[4]:
            h4 = torch.cat((h4, self.context_params.expand(h4.size(0), -1)), dim=1)

        y = self.fc1(h4)

        return y
