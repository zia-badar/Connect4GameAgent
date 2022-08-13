import torch.nn
from torch import nn
from torch.nn import Parameter
from torch.nn.functional import one_hot

from environement import C4Env


class Policy(nn.Module):           # only support discret single actions

    def __init__(self, state_space, action_space):
        super(Policy, self).__init__()
        input_size = torch.prod(torch.tensor([list(state_space)]), 1)
        output_size = torch.prod(torch.tensor([list(action_space)]), 1)
        self.head = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(4, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(72, 16)
        )

        self.action_layer = nn.Linear(16, output_size)
        self.value_layer = nn.Linear(16, 1)

    # https://stats.stackexchange.com/questions/64081/how-do-i-sample-from-a-discrete-categorical-distribution-in-log-space
    def act(self, x):
        x = x.unsqueeze(0).unsqueeze(0)     # converting to batch x channel x height x width format
        h = self.head(x)
        al = self.action_layer(h)
        a = torch.distributions.Categorical(logits=al).sample()
        return a

    def value(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)         # adding batch and channel dim
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)                    # adding channel dim
        h = self.head(x)
        v = self.value_layer(h)
        return v

    def prob(self, a, s):
        s = s.unsqueeze(1)          # adding channel dimension
        h = self.head(s)
        al = self.action_layer(h)
        p_x = torch.softmax(al, dim=1)
        p_y = one_hot(a, num_classes=C4Env.board_size[1])
        nlogp = -torch.sum(p_y * torch.log(p_x), dim=1)
        p = 1/torch.exp(nlogp)
        return p

    def entropy(self, s):
        s = s.unsqueeze(1)          # adding channel dimension
        h = self.head(s)
        al = self.action_layer(h)
        p = torch.softmax(al, dim=1)
        e = -torch.sum(p * torch.log(p), dim=1)
        return e