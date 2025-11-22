import torch
from torch import nn

class MessagePassingBase(nn.Module):
    """
    Minimal base so that layer implementations calling:
        - message_and_aggregate(graph, input)
        - combine(input, update)
    work as expected.

    Forward calls the two phases and returns combine output.
    """
    def __init__(self):
        super(MessagePassingBase, self).__init__()

    def forward(self, graph, input):
        # Layers in your repo may pass different arglists; we attempt flexible call
        update = self.message_and_aggregate(graph, input)
        return self.combine(input, update)

class MLP(nn.Module):
    def __init__(self, input_dim, dims):
        super(MLP, self).__init__()
        layers = []
        cur = input_dim
        for d in dims:
            layers.append(nn.Linear(cur, d))
            layers.append(nn.ReLU())
            cur = d
        # remove final activation
        if layers:
            layers = layers[:-1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
