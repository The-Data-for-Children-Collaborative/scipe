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
    
class PairLayer(torch.nn.Module):
    """ Layer that returns pair of outputs from models on forward pass.

        Attributes:
            model1 (torch.nn.Module): Model for first output of pair.
            model2 (torch.nn.Module): Model for second output of pair.
    """
    def __init__(self, model1, model2):
        super(PairLayer, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
    def __getitem__(self, key):
        if key == 0:
            return self.model1
        elif key == 1:
            return self.model2
        else:
            raise IndexError('Pair index out of range')

    def forward(self, x):
        return self.model1(x), self.model2(x)
