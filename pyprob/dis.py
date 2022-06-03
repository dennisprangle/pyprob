# Code to implement DIS algorithm
from tarfile import FIFOTYPE
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from . import InferenceEngine, Model, TraceMode
from .distributions import Empirical
from pyprob.util import InferenceEngine, InferenceNetwork

def effective_sample_size(w):
    """Effective sample size of weights

    `w` is a 1-dimensional tensor of weights (normalised or unnormalised)"""
    sumw = torch.sum(w)
    if sumw == 0:
        return 0.

    return (sumw ** 2.0) / torch.sum(w ** 2.0)


def get_alternate_weights(sqd, old_weights, old_eps, new_eps):
    """Return weights appropriate to another `epsilon` value"""
    # Interpretable version of the generic reweighting code:
    # w = old_weights
    # d = torch.sqrt(sqd)
    # w /= torch.exp(-0.5*(d / old_eps)**2.)
    # w *= torch.exp(-0.5*(d / new_eps)**2.)
    # w /= sum(w)

    w = old_weights.detach().clone()
    if new_eps == 0:
        # Remove existing distance-based weight contribution
        w /= torch.exp(-0.5 * sqd / old_eps**2.)
        # Replace with indicator function weight contribution
        w = torch.where(
            sqd==0.,
            w,
            torch.zeros_like(w)
        )
    else:
        # An efficient way to do the generic case
        a = 0.5 * (old_eps**-2. - new_eps**-2.)
        w *= torch.exp(sqd*a)

    sumw = torch.sum(w)
    if sumw > 0.:
        w /= sumw
    return w


def find_eps(sqd, old_weights, old_eps, target_ess, upper, bisection_its=50):
        """Return epsilon value <= `upper` giving ess matching `target_ess` as closely as possible

        Bisection search is performed using `bisection_its` iterations
        """
        w = get_alternate_weights(sqd, old_weights, old_eps, upper)
        ess = effective_sample_size(w)
        if ess < target_ess:
            return upper

        lower = 0.
        for i in range(bisection_its):
            eps_guess = (lower + upper) / 2.
            w = get_alternate_weights(sqd, old_weights, old_eps, eps_guess)
            ess = effective_sample_size(w)
            if ess > target_ess:
                upper = eps_guess
            else:
                lower = eps_guess

        # Consider returning eps=0 if it's still an endpoint
        if lower == 0.:
            w = get_alternate_weights(sqd, old_weights, old_eps, 0.)
            ess = effective_sample_size(w)
            if ess > target_ess:
                return 0.

        # Be conservative by returning upper end of remaining range
        return upper


class ModelDIS(Model):
    def __init__(self, epsilon=np.inf, obs = None, dist_fun = None, weight_truncation=False,**kwargs):
        super().__init__(kwargs)
        self.dis_model = True
        self.epsilon = epsilon
        self.distances = None
        self.dist_fun = dist_fun
        self.obs = obs
        self.ess = 0
        self.weight_truncation = weight_truncation

    # def _traces(self, *args, **kwargs):
    #     # Set observable values whenever _traces is run, even from 'prior' or 'posterior'
    #     # (allows prior and posterior to create samples suitable for training)
    #     kwargs['trace_mode'] = TraceMode.PRIOR_FOR_INFERENCE_NETWORK
    #     return super()._traces(*args, **kwargs)

    def update_DIS_weights(self, posterior, ess_target = 500):
        # Modify weights to take distance into account
        if not self.dist_fun:
            raise RuntimeError('Cannot extract distances. Ensure the model is initialised with a distance measure: dist_fun = ... ')
        # Consider future efficiency: broadcasting / working with posterior_results etc.


        def distance(trace):
            #trace_obs = [v for v in trace.variables_observed if v.name != 'dummy']
            trace_obs = [v.value for v in trace.variables_observed if v.name != 'dummy']
            return self.dist_fun(self.obs, trace_obs)
        self.distances = torch.tensor([distance(x) for x in posterior.values])
        posterior.sqd = self.distances**2



        log_w_contrib = -0.5 * posterior.sqd / self.epsilon**2.
        # Note - can't set posterior.weights directly (it's a property with no setter)
        # But can access the underlying variables in which weights is stored.
        posterior._categorical.logits += log_w_contrib
        posterior._categorical.probs = np.exp(posterior._categorical.logits)

        # Update epsilon 
        w = posterior.weights
        
        if self.epsilon == np.inf:
            # A finite value needed, so pick a sensible upper bound
            upper_eps = torch.max(posterior.sqd).item()
        else:
            upper_eps = self.epsilon

        new_epsilon = find_eps(posterior.sqd, w, self.epsilon, ess_target, upper_eps)
        w = get_alternate_weights(posterior.sqd, w, self.epsilon, new_epsilon)
        self.epsilon = new_epsilon
        # self.ess = posterior._effective_sample_size # Should set this maybe...
        self.ess = effective_sample_size(w)
        posterior.dis_eps = self.epsilon

        return posterior

    # def learn_inference_network(self, num_traces, num_traces_end=1000000000, inference_engine=None, importance_sample_size=None, inference_network=..., prior_inflation=..., dataset_dir=None, dataset_valid_dir=None, observe_embeddings=..., batch_size=64, valid_size=None, valid_every=None, optimizer_type=..., learning_rate_init=0.001, learning_rate_end=0.000001, learning_rate_scheduler_type=..., momentum=0.9, weight_decay=0, save_file_name_prefix=None, save_every_sec=600, pre_generate_layers=False, distributed_backend=None, distributed_params_sync_every_iter=10000, distributed_num_buckets=None, dataloader_offline_num_workers=0, stop_with_bad_loss=True, log_file_name=None, lstm_dim=512, lstm_depth=1, proposal_mixture_components=10):
    #     return super().learn_inference_network(num_traces, num_traces_end, inference_engine, importance_sample_size, inference_network, prior_inflation, dataset_dir, dataset_valid_dir, observe_embeddings, batch_size, valid_size, valid_every, optimizer_type, learning_rate_init, learning_rate_end, learning_rate_scheduler_type, momentum, weight_decay, save_file_name_prefix, save_every_sec, pre_generate_layers, distributed_backend, distributed_params_sync_every_iter, distributed_num_buckets, dataloader_offline_num_workers, stop_with_bad_loss, log_file_name, lstm_dim, lstm_depth, proposal_mixture_components)

    def train(self, iterations=10, num_traces = 500, importance_sample_size=5000, ess_target=500, batch_size=100, num_workers = 1, **kwargs):
        for i in range(iterations):
            # TO DO: suppress messages about OfflineDataset creation
            self.learn_inference_network(
                num_traces,
                importance_sample_size=importance_sample_size,
                ess_target = ess_target,
                inference_engine=InferenceEngine.DISTILLING_IMPORTANCE_SAMPLING,
                batch_size=batch_size,
                observe_embeddings={"dummy":{'dim':1, 'depth':1}},
                num_workers = num_workers,
                inference_network=InferenceNetwork.LSTM,
                **kwargs
                )

            # TO DO: Improve reporting results?
            print(f"Training iterations {i+1} "
                  f" epsilon {self.epsilon:.2f} "
                  f" ESS {self.ess:.1f}")

    def save(self, file_name):
        self.save_inference_network(file_name)
