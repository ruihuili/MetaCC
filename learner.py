import copy
from collections import OrderedDict

import torch
import torch.nn as nn


def maml_init_(module):
    torch.nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    torch.nn.init.constant_(module.bias.data, 0.0)
    return module


class Learner(nn.Module):

    def __init__(self, image_size):
        super(Learner, self).__init__()
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 64, (3, 3), stride=(2, 2), padding=1, bias=True)),
            ('norm1', nn.BatchNorm2d(64, affine=True)),
            ('relu1', nn.ReLU()),

            ('conv2', nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1, bias=True)),
            ('norm2', nn.BatchNorm2d(64, affine=True)),
            ('relu2', nn.ReLU()),

            ('conv3', nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1, bias=True)),
            ('norm3', nn.BatchNorm2d(64, affine=True)),
            ('relu3', nn.ReLU()),

            ('conv4', nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1, bias=True)),
            ('norm4', nn.BatchNorm2d(64, affine=True)),
            ('relu4', nn.ReLU())]))
        })
        self.model.update({'cls': nn.Linear(64, 10, bias=True)})

        # initialize weights
        maml_init_(self.model['features'].conv1)
        torch.nn.init.uniform_(self.model['features'].norm1.weight)
        maml_init_(self.model['features'].conv2)
        torch.nn.init.uniform_(self.model['features'].norm2.weight)
        maml_init_(self.model['features'].conv3)
        torch.nn.init.uniform_(self.model['features'].norm3.weight)
        maml_init_(self.model['features'].conv4)
        torch.nn.init.uniform_(self.model['features'].norm4.weight)
        maml_init_(self.model['cls'])

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.model.features(x)
        x = torch.reshape(x, [x.size(0), -1])
        outputs = self.model.cls(x)
        return outputs

    def get_flat_params(self):
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def copy_flat_params(self, cI):
        idx = 0
        for p in self.model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def transfer_params(self, learner_w_grad, cI):
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen

    def reset_batch_stats(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

