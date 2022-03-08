import torch

from . import Distribution
from .. import util


class Delta(Distribution):
    def __init__(self, delta_val):
        super().__init__(name='Delta', address_suffix='Delta')
        self._delta_val = util.to_tensor(delta_val)
        if delta_val.dim() == 0 or delta_val.dim() == 1:
            self._batch_length = 1
        else: 
            self._batch_length = delta_val.shape[0] #Does this even make sense? Leaves are 2d obs by default so would be wrong.

    def __repr__(self):
        return 'Delta({})'.format(self.delta_val.detach().cpu().numpy().tolist())

    
    def sample(self):
        return self.delta_val

    def log_prob(self, value, sum = False): # Not sure what point of sum is... need to be careful with dimensions here.
        lp = torch.log(self.prob(value))
        return lp if sum else lp.sum()
        #return torch.zeros(self._batch_length).unsqueeze(-1)

    def prob(self, value):
        return (self.delta_val == value).type(torch.float64)
        #return torch.zeros(self._batch_length).unsqueeze(-1)
    
    @property
    def delta_val(self):
        return self._delta_val
    
    @property
    def mean(self):
        return self.delta_val
    
    @property
    def variance(self):
        return torch.zeros(self._batch_length).unsqueeze(-1)

    def to(self, device):
        return Delta(self.delta_val.to(device))
