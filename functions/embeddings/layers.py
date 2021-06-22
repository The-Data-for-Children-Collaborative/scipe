import torch


class LambdaLayer(torch.nn.Module):
    """ Layer that applies arbitrary function in forward pass.

        Attributes:
            lambd (function): Function to apply in forward pass.
    """
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
