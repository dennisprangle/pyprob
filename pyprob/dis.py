# Code to implement DIS algorithm
from tarfile import FIFOTYPE
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from . import InferenceEngine, Model, TraceMode
from .distributions import Empirical

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
    def __init__(self, epsilon=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _traces(self, *args, **kwargs):
        # Set observable values whenever _traces is run, even from 'prior' or 'posterior'
        # (allows prior and posterior to create samples suitable for training)
        kwargs['trace_mode'] = TraceMode.PRIOR_FOR_INFERENCE_NETWORK
        return super()._traces(*args, **kwargs)


    def update_DIS_posterior_weights(self, posterior):
        # Modify weights to take distance into account
        try:
            dist = [
                x.named_variables["distance"].value.item()
                for x in posterior.values
            ]
            sqd = torch.tensor(dist) ** 2.
        except:
            raise RuntimeError('Cannot extract distances. Ensure the "forward" method computes the distance between the simulation and an observed dataset, and stores it as a pyprob observation named "distance".')
        log_w_contrib = -0.5 * sqd / self.epsilon**2.
        posterior.sqd = sqd
        # Note - can't set posterior.weights directly (it's a property with no setter)
        # But can access the underlying variables in which weights is stored.
        posterior._categorical.probs = posterior.weights * torch.exp(log_w_contrib)
        return posterior


    def train(self, iterations=10, importance_sample_size=1000, ess_target=500, batch_size=100, nbatches=5, **kwargs):
        for i in range(iterations):
            if self._inference_network is None:
                sample = self.prior(
                    num_traces=importance_sample_size,
                )
            else:
                sample = self.posterior(
                    num_traces=importance_sample_size,
                    inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK,
                    observe={"bool_func": 1} # TO DO: remove need for this to be hardcoded in (e.g. create inference network without observation)
                )
            sample = self.update_DIS_posterior_weights(sample)
            w = sample.weights
            sqd = sample.sqd
            upper_eps = self.epsilon
            if upper_eps == np.inf:
                # A finite value needed, so pick a sensible upper bound
                upper_eps = torch.max(sqd).item()
            new_epsilon = find_eps(sqd, w, self.epsilon, ess_target, upper_eps)
            w = get_alternate_weights(sqd, w, self.epsilon, new_epsilon)
            self.epsilon = new_epsilon
            ess = effective_sample_size(w)
            # TO DO: TRUNCATE WEIGHTS?
            # Delete existing files so results not just appended to old ones
            file_name = "pyprob_traces_training_batch"
            if os.path.exists(file_name):
                os.remove(file_name)
            if os.path.exists("pyprob_hashes"):
                os.remove("pyprob_hashes")
            batch = Empirical(file_name=file_name)
            # Resample importance sample to get a training batch
            # TO DO: In the longer term, maybe allow 'learn_inference_network'
            # to cope with a weighted training batch. I think this requires
            # adding a sampler handling weights to 'inference_network.optimize')
            indices = torch.multinomial(w, batch_size*nbatches, replacement=True)
            for j in indices:
                batch.add(sample.values[j], 0.)
            batch._shelf['__length'] = batch_size*nbatches
            batch.close()
            # TO DO: suppress messages about OfflineDataset creation
            self.learn_inference_network(
                num_traces=nbatches*batch_size, # TO DO: is this correct???
                dataset_dir=".",
                batch_size=batch_size,
                **kwargs
            )
            # TO DO: Improve reporting results?
            print(f"Training iterations {i+1} "
                  f" epsilon {self.epsilon:.2f} "
                  f" ESS {ess:.1f}")

    def save(self, file_name):
        self.save_inference_network(file_name)

