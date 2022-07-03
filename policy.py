import torch.nn
from torch.nn import Parameter

from environement import C4Env


class Policy:           # only support discret single actions

    def __init__(self, state_space, action_space):
        input_size = torch.prod(torch.tensor([list(state_space)]), 1)
        output_size = torch.prod(torch.tensor([list(action_space)]), 1)
        self.layer1 = torch.nn.Linear(input_size, 512)
        self.layer2 = torch.nn.Linear(512, output_size)
        self.valueLayer = torch.nn.Linear(512, 1)

    # https://stats.stackexchange.com/questions/64081/how-do-i-sample-from-a-discrete-categorical-distribution-in-log-space
    def act(self, x):
        x = torch.nn.Flatten()(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.argmax(x - torch.log(-torch.log(torch.rand(x.shape))), dim=1)
        return x

    def value(self, x):
        x = torch.nn.Flatten()(x)
        x = self.layer1(x)
        x = self.valueLayer(x)
        return x

    def prob(self, a, s):
        a = torch.nn.functional.one_hot(a, num_classes = C4Env.board_size[1])
        s = torch.nn.Flatten()(s)
        _a = self.layer1(s)
        _a = self.layer2(_a)
        _a = torch.softmax(_a, dim=1)
        nlogp = -torch.sum(a * torch.log(_a), dim=1)
        return 1/torch.exp(nlogp)

    def entropy(self, s):
        x = torch.nn.Flatten()(s)
        x = self.layer1(x)
        x = self.layer2(x)
        p = torch.softmax(x, dim=1)
        e = -torch.sum(p * torch.log(p), dim=-1)
        return e

    def paramters(self):
        return [self.layer1.weight, self.layer2.weight, self.valueLayer.weight]
