# Some annotations by Sammy (occasional speculation)

from py import process
import torch
import time
import sys
import os
import math
import random
import warnings
import numpy as np
from copy import deepcopy
from termcolor import colored
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

from .distributions import Empirical
from . import util, state, TraceMode, PriorInflation, InferenceEngine, InferenceNetwork, Optimizer, LearningRateScheduler, AddressDictionary
from .nn import InferenceNetwork as InferenceNetworkBase
from .nn import OnlineDataset, OfflineDataset, InferenceNetworkFeedForward, InferenceNetworkLSTM
from .remote import ModelServer

trace_list = mp.Manager().list()
counter = mp.Value('I',0)


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


def unwrap_make_trace(self, index, *args, **kwargs):
    print('entering unwrap')
    util.seed()
    state._begin_trace()
    result = self.forward(*args, **kwargs)
    trace = state._end_trace(result)
    trace_list[index] = (trace, trace.log_prob)
    with counter.get_lock():
        counter.value += 1
        print(counter.value)
    
class Parallel_Generator(Dataset):
    """
    Generates datasets for parallelisation by PyTorch dataloader methods.
    """
    def __init__(self, model, importance_sample_size = None, observe = None, *args, **kwargs):
        self._model = model
        self._generator = model._trace_generator
        self._inference_network = model._inference_network
        self._length = importance_sample_size
        self._observe = observe
        # self._args = args
        # self._kwargs = kwargs
        # if self._model._inference_network:
        #     state._init_traces(func=self._model.forward, trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, inference_network=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe=observe)
        # else:
        #     state._init_traces(func=self._model.forward,trace_mode=TraceMode.PRIOR)

    
    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # state._begin_trace()
        # result = self._model.forward(*self._args, **self._kwargs)
        # trace = state._end_trace(result)
        # return trace
        if self._model._inference_network:
            return next(self._generator(trace_mode=TraceMode.POSTERIOR, inference_engine = InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, inference_network = self._inference_network, observe = self._observe))
        else:
            return next(self._generator(trace_mode=TraceMode.PRIOR))


class Model():
    def __init__(self, name='Unnamed PyProb model', address_dict_file_name=None, epsilon=np.inf, obs = None, dist_fun = None, **kwargs):
        super().__init__()
        self.name = name
        self.dis_model = False
        self.epsilon = epsilon
        self.distances = None
        self.dist_fun = dist_fun
        self.obs = obs
        self.ess = 0
        self._inference_network = None
        if address_dict_file_name is None:
            self._address_dictionary = None
        else:
            self._address_dictionary = AddressDictionary(address_dict_file_name)

    def __repr__(self):
        return 'Model(name:{})'.format(self.name)

    def forward(self):
        raise RuntimeError('Model instances must provide a forward method.')

    # Each yield statement generates the trace of one run of the simulator.
    # Trace should include values of sample statements, their addresses and call numbers.
    def _trace_generator(self, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, observe=None, metropolis_hastings_trace=None, likelihood_importance=1., *args, **kwargs):
        state._init_traces(func=self.forward, trace_mode=trace_mode, prior_inflation=prior_inflation, inference_engine=inference_engine, inference_network=inference_network, observe=observe, metropolis_hastings_trace=metropolis_hastings_trace, address_dictionary=self._address_dictionary, likelihood_importance=likelihood_importance)
        while True:
            state._begin_trace()
            result = self.forward(*args, **kwargs)
            trace = state._end_trace(result)
            yield trace


    def _dis_traces(self, num_traces=5000, trace_mode=TraceMode.POSTERIOR, inference_engine=InferenceEngine.DISTILLING_IMPORTANCE_SAMPLING, ess_target=500, map_func=None, silent=False, observe=None, file_name=None, likelihood_importance=1., num_workers=1, *args, **kwargs):
        
              
        if not self.dist_fun:
            raise RuntimeError('Cannot extract distances. Ensure the model is initialised with a distance measure: dist_fun = ... ')

        if self._inference_network is None:
            warnings.warn('No inference network found. Sampling from prior')
        #     if num_workers > 1:
        #         warnings.warn('Not parallelising, since it normally does not improve speed of sampling from prior')
        #     traces = self._traces(num_traces=num_traces, trace_mode=TraceMode.PRIOR_FOR_INFERENCE_NETWORK, inference_engine=InferenceEngine.DISTILLING_IMPORTANCE_SAMPLING, inference_network=None, map_func=map_func, silent=silent, observe=None, file_name=file_name, likelihood_importance=likelihood_importance, *args, **kwargs)
        #     return traces

        # if num_workers <=1:
        #     return self._traces(num_traces=num_traces, trace_mode=trace_mode, inference_engine=InferenceEngine.DISTILLING_IMPORTANCE_SAMPLING, inference_network=self._inference_network, map_func=map_func, silent=silent, observe=observe, file_name=file_name, likelihood_importance=likelihood_importance, *args, **kwargs)

        if map_func is None: # Some function of the trace... e.g. summary of data?
            map_func = lambda trace: trace
        
        sqdists = []
        log_weights = util.to_tensor(torch.zeros(num_traces))
        traces = Empirical(file_name=file_name)

        if trace_mode == TraceMode.PRIOR or trace_mode == TraceMode.PRIOR_FOR_INFERENCE_NETWORK:
            observe = None
            
        state._init_traces(func=self.forward, trace_mode=trace_mode, inference_engine=inference_engine, inference_network=self._inference_network, observe=observe, address_dictionary=self._address_dictionary)
        time_start = time.time()
        if (util._verbosity > 1) and not silent: # Logs progress.
            len_str_num_traces = len(str(num_traces))
            print('Time spent  | Time remain.| Progress             | {} | {} | Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1), 'ESS'.ljust(len_str_num_traces+2)))
            prev_duration = 0
        chunk_size, remainder = divmod(num_traces, num_workers)


        if not remainder:
            chunks = [range(i*chunk_size, (i+1)*chunk_size) for i in range(num_workers)]
        else:
            chunk_size = num_traces // (num_workers)
            chunks = [range(i*chunk_size, (i+1)*chunk_size) for i in range(num_workers-1)]
            chunks.append(range(chunk_size*(num_workers-1), num_traces))

        def distance(trace):
            #trace_obs = [v for v in trace.variables_observed if v.name != 'dummy']
            trace_obs = [v.value for v in trace.variables_observed if v.name != 'dummy']
            return self.dist_fun(self.obs, trace_obs)

        def make_trace(chunk, q):
            util.seed()
            #L = []
            for i in chunk:
                state._begin_trace()
                result = self.forward(*args, **kwargs)
                trace = state._end_trace(result)
                sqd = distance(trace)**2
                q.put((map_func(trace), trace.log_importance_weight, sqd))

        if num_workers == 1 or trace_mode == TraceMode.PRIOR or trace_mode == trace_mode.PRIOR_FOR_INFERENCE_NETWORK:
            generator = self._trace_generator(trace_mode=trace_mode, inference_engine=inference_engine, observe=observe, likelihood_importance=likelihood_importance, *args, **kwargs)
            for i in range(num_traces):

                trace = next(generator)
                
                if trace_mode == TraceMode.PRIOR or trace_mode == TraceMode.PRIOR_FOR_INFERENCE_NETWORK:
                    log_weight = 1
                else:
                    log_weight = trace.log_importance_weight

                
                if util.has_nan_or_inf(log_weight):
                    warnings.warn('Encountered trace with nan, inf, or -inf log_weight. Discarding trace.')
                    if i > 0:
                        log_weights[i] = log_weights[-1]
                else:
                    traces.add(trace, log_weight)
                    sqd = distance(trace)**2
                    sqdists.append(sqd)
                    log_weights[i] = log_weight

                if (util._verbosity > 1) and not silent:
                    duration = time.time() - time_start
                    if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                        prev_duration = duration
                        traces_per_second = (i + 1) / duration
                        effective_sample_size_track = float(1./torch.distributions.Categorical(logits=log_weights[:i+1]).probs.pow(2).sum())
                        if util.has_nan_or_inf(effective_sample_size_track):
                            effective_sample_size_track = 0
                        print('{} | {} | {} | {}/{} | {} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, '{:.2f}'.format(effective_sample_size_track).rjust(len_str_num_traces+2), traces_per_second), end='\r')
                        if i < num_traces - 1:
                            sys.stdout.flush()


        else:

            processes = []
            queue = mp.Queue()

            for i in range(num_workers):
                p = mp.Process(target=make_trace, args=(chunks[i], queue))
                processes.append(p)  

            for p in processes:
                p.start()

            for i in range(num_traces):
                trace, log_weight, sqd = deepcopy(queue.get())
                if util.has_nan_or_inf(log_weight):
                    warnings.warn('Encountered trace with nan, inf, or -inf log_weight. Discarding trace.')
                    if i > 0:
                        log_weights[i] = log_weights[-1]
                else:
                    traces.add(trace, log_weight)
                    sqdists.append(sqd)
                    log_weights[i] = log_weight

                if (util._verbosity > 1) and not silent:
                    duration = time.time() - time_start
                    if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                        prev_duration = duration
                        traces_per_second = (i + 1) / duration
                        effective_sample_size_track = float(1./torch.distributions.Categorical(logits=log_weights[:i+1]).probs.pow(2).sum())
                        if util.has_nan_or_inf(effective_sample_size_track):
                            effective_sample_size_track = 0
                        print('{} | {} | {} | {}/{} | {} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, '{:.2f}'.format(effective_sample_size_track).rjust(len_str_num_traces+2), traces_per_second), end='\r')
                        if i < num_traces - 1:
                            sys.stdout.flush()

            for p in processes:
                p.join()
                p.close()

        if (util._verbosity > 1) and not silent:
            print()
        # for trace_list in trace_batches:
        #     for trace, log_prob in trace_list:
        #         traces.add(trace, log_prob)
        

        traces.log_weights = util.to_tensor(traces.log_weights)
        w = torch.exp(traces.log_weights)
        sqdists = util.to_tensor(sqdists)
        upper_eps = self.epsilon

        if upper_eps == np.inf:
            # A finite value needed, so pick a sensible upper bound
            upper_eps = torch.max(sqdists).item()
        new_epsilon = find_eps(sqdists, w, self.epsilon, ess_target, upper_eps)
        w = get_alternate_weights(sqdists, w, self.epsilon, new_epsilon)
        self.epsilon = new_epsilon
        # self.ess = posterior._effective_sample_size # Should set this maybe...
        self.ess = effective_sample_size(w)
        traces.dis_eps=new_epsilon

        log_w_contrib = -0.5 * sqdists / self.epsilon**2.
        traces.log_weights += log_w_contrib

        traces.finalize()


        return traces


    #Invokes the trace generator.
    def _traces(self, num_traces=10, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, map_func=None, silent=False, observe=None, file_name=None, likelihood_importance=1., *args, **kwargs):
        generator = self._trace_generator(trace_mode=trace_mode, prior_inflation=prior_inflation, inference_engine=inference_engine, inference_network=inference_network, observe=observe, likelihood_importance=likelihood_importance, *args, **kwargs)
        traces = Empirical(file_name=file_name) # From file.
        if map_func is None: # Some function of the trace... e.g. summary of data?
            map_func = lambda trace: trace
        log_weights = util.to_tensor(torch.zeros(num_traces)) # Initialisation -> weights are 1, logs 0.
        time_start = time.time()
        if (util._verbosity > 1) and not silent: # Logs progress.
            len_str_num_traces = len(str(num_traces))
            print('Time spent  | Time remain.| Progress             | {} | {} | Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1), 'ESS'.ljust(len_str_num_traces+2)))
            prev_duration = 0


        for i in range(num_traces):
            trace = next(generator)
            if trace_mode == TraceMode.PRIOR:
                log_weight = 1. # Log weight, should probably be 0. 
            else:
                log_weight = trace.log_importance_weight
            if util.has_nan_or_inf(log_weight):
                warnings.warn('Encountered trace with nan, inf, or -inf log_weight. Discarding trace.')
                if i > 0:
                    log_weights[i] = log_weights[-1]
            else:
                traces.add(map_func(trace), log_weight)
                log_weights[i] = log_weight

            if (util._verbosity > 1) and not silent:
                duration = time.time() - time_start
                if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                    prev_duration = duration
                    traces_per_second = (i + 1) / duration
                    effective_sample_size = float(1./torch.distributions.Categorical(logits=log_weights[:i+1]).probs.pow(2).sum())
                    if util.has_nan_or_inf(effective_sample_size):
                        effective_sample_size = 0
                    print('{} | {} | {} | {}/{} | {} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, '{:.2f}'.format(effective_sample_size).rjust(len_str_num_traces+2), traces_per_second), end='\r')
                    sys.stdout.flush()

        if (util._verbosity > 1) and not silent:
            print()
        traces.finalize()
        return traces

    def get_trace(self, *args, **kwargs):
        warnings.warn('Model.get_trace will be deprecated in future releases. Use Model.sample instead.')
        return next(self._trace_generator(*args, **kwargs))

    def sample(self, *args, **kwargs):
        return next(self._trace_generator(*args, **kwargs))

    def prior(self, num_traces=10, prior_inflation=PriorInflation.DISABLED, map_func=None, file_name=None, likelihood_importance=1., *args, **kwargs):
        prior = self._traces(num_traces=num_traces, trace_mode=TraceMode.PRIOR, prior_inflation=prior_inflation, map_func=map_func, file_name=file_name, likelihood_importance=likelihood_importance, *args, **kwargs)
        prior.rename('Prior, traces: {:,}'.format(prior.length))
        prior.add_metadata(op='prior', num_traces=num_traces, prior_inflation=str(prior_inflation), likelihood_importance=likelihood_importance)
        return prior

    def prior_results(self, num_traces=10, prior_inflation=PriorInflation.DISABLED, map_func=lambda trace: trace.result, file_name=None, likelihood_importance=1., *args, **kwargs):
        return self.prior(num_traces=num_traces, prior_inflation=prior_inflation, map_func=map_func, file_name=file_name, likelihood_importance=likelihood_importance, *args, **kwargs)

    def posterior(self, num_traces=10, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, initial_trace=None, map_func=None, observe=None, file_name=None, thinning_steps=None, likelihood_importance=1., *args, **kwargs):
        if inference_engine == InferenceEngine.IMPORTANCE_SAMPLING:
            posterior = self._traces(num_traces=num_traces, trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, inference_network=None, map_func=map_func, observe=observe, file_name=file_name, likelihood_importance=likelihood_importance, *args, **kwargs)
            posterior.rename('Posterior, IS, traces: {:,}, ESS: {:,.2f}'.format(posterior.length, posterior.effective_sample_size))
            posterior.add_metadata(op='posterior', num_traces=num_traces, inference_engine=str(inference_engine), effective_sample_size=posterior.effective_sample_size, likelihood_importance=likelihood_importance)
        elif inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            if self._inference_network is None:
                raise RuntimeError('Cannot run inference engine IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK because no inference network for this model is available. Use learn_inference_network or load_inference_network first.')
            with torch.no_grad():
                posterior = self._traces(num_traces=num_traces, trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, inference_network=self._inference_network, map_func=map_func, observe=observe, file_name=file_name, likelihood_importance=likelihood_importance, *args, **kwargs)
            posterior.rename('Posterior, IC, traces: {:,}, train. traces: {:,}, ESS: {:,.2f}'.format(posterior.length, self._inference_network._total_train_traces, posterior.effective_sample_size))
            posterior.add_metadata(op='posterior', num_traces=num_traces, inference_engine=str(inference_engine), effective_sample_size=posterior.effective_sample_size, likelihood_importance=likelihood_importance, train_traces=self._inference_network._total_train_traces)
        elif inference_engine == InferenceEngine.DISTILLING_IMPORTANCE_SAMPLING:
            if self._inference_network is None:
                warnings.warn('No inference network specified. Sampling from prior instead.')
                # Note: Observe = None in below. This is necessary so that init_traces correctly functions and observables are correctly embedded.
                posterior = self._traces(num_traces=num_traces, trace_mode=TraceMode.PRIOR_FOR_INFERENCE_NETWORK, inference_engine=inference_engine, inference_network=self._inference_network, map_func=map_func, observe=None, file_name=file_name, likelihood_importance=likelihood_importance, *args, **kwargs)
                posterior.rename('Prior, traces: {:,}'.format(posterior.length))
                posterior.add_metadata(op='prior', num_traces=num_traces, likelihood_importance=likelihood_importance)
            else: 
                with torch.no_grad():
                    posterior = self._traces(num_traces=num_traces, trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, inference_network=self._inference_network, map_func=map_func, observe=observe, file_name=file_name, likelihood_importance=likelihood_importance, *args, **kwargs)
                    posterior.rename('Posterior, IC, traces: {:,}, train. traces: {:,}, ESS: {:,.2f}'.format(posterior.length, self._inference_network._total_train_traces, posterior.effective_sample_size))
                    posterior.add_metadata(op='posterior', num_traces=num_traces, inference_engine=str(inference_engine), effective_sample_size=posterior.effective_sample_size, likelihood_importance=likelihood_importance, train_traces=self._inference_network._total_train_traces)
        else:  # inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS
            posterior = Empirical(file_name=file_name)
            if map_func is None:
                map_func = lambda trace: trace
            if initial_trace is None:
                initial_trace = next(self._trace_generator(trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, observe=observe, *args, **kwargs))
            if len(initial_trace) == 0:
                raise RuntimeError('Cannot run MCMC inference with empty initial trace. Make sure the model has at least one pyprob.sample statement.')

            current_trace = initial_trace

            time_start = time.time()
            traces_accepted = 0
            samples_reused = 0
            samples_all = 0
            if thinning_steps is None:
                thinning_steps = 1

            if util._verbosity > 1:
                len_str_num_traces = len(str(num_traces))
                print('Time spent  | Time remain.| Progress             | {} | Accepted|Smp reuse| Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1)))
                prev_duration = 0

            for i in range(num_traces):
                if util._verbosity > 1:
                    duration = time.time() - time_start
                    if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                        prev_duration = duration
                        traces_per_second = (i + 1) / duration
                        print('{} | {} | {} | {}/{} | {} | {} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, '{:,.2f}%'.format(100 * (traces_accepted / (i + 1))).rjust(7), '{:,.2f}%'.format(100 * samples_reused / max(1, samples_all)).rjust(7), traces_per_second), end='\r')
                        sys.stdout.flush()
                candidate_trace = next(self._trace_generator(trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, metropolis_hastings_trace=current_trace, observe=observe, *args, **kwargs))

                log_acceptance_ratio = math.log(current_trace.length_controlled) - math.log(candidate_trace.length_controlled) + candidate_trace.log_prob_observed - current_trace.log_prob_observed
                for variable in candidate_trace.variables_controlled:
                    if variable.reused:
                        log_acceptance_ratio += torch.sum(variable.log_prob)
                        log_acceptance_ratio -= torch.sum(current_trace.variables_dict_address[variable.address].log_prob)
                        samples_reused += 1
                samples_all += candidate_trace.length_controlled

                if state._metropolis_hastings_site_transition_log_prob is None:
                    warnings.warn('Trace did not hit the Metropolis Hastings site, ensure that the model is deterministic except pyprob.sample calls')
                else:
                    log_acceptance_ratio += torch.sum(state._metropolis_hastings_site_transition_log_prob)

                # print(log_acceptance_ratio)
                if math.log(random.random()) < float(log_acceptance_ratio):
                    traces_accepted += 1
                    current_trace = candidate_trace
                # do thinning
                if i % thinning_steps == 0:
                    posterior.add(map_func(current_trace))

            if util._verbosity > 1:
                print()

            posterior.finalize()
            posterior.rename('Posterior, {}, traces: {:,}{}, accepted: {:,.2f}%, sample reuse: {:,.2f}%'.format('LMH' if inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS else 'RMH', posterior.length, '' if thinning_steps == 1 else ' (thinning steps: {:,})'.format(thinning_steps), 100 * (traces_accepted / num_traces), 100 * samples_reused / samples_all))
            posterior.add_metadata(op='posterior', num_traces=num_traces, inference_engine=str(inference_engine), likelihood_importance=likelihood_importance, thinning_steps=thinning_steps, num_traces_accepted=traces_accepted, num_samples_reuised=samples_reused, num_samples=samples_all)
        return posterior

    def posterior_results(self, num_traces=10, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, initial_trace=None, map_func=lambda trace: trace.result, observe=None, file_name=None, thinning_steps=None, *args, **kwargs):
        return self.posterior(num_traces=num_traces, inference_engine=inference_engine, initial_trace=initial_trace, map_func=map_func, observe=observe, file_name=file_name, thinning_steps=thinning_steps, *args, **kwargs)

    def reset_inference_network(self):
        self._inference_network = None

    def learn_inference_network(self, num_traces, num_traces_end=1e9, inference_engine = None, importance_sample_size = None, ess_target=500, inference_network=InferenceNetwork.FEEDFORWARD, prior_inflation=PriorInflation.DISABLED, dataset_dir=None, dataset_valid_dir=None, observe_embeddings={}, batch_size=64, valid_size=None, valid_every=None, optimizer_type=Optimizer.ADAM, learning_rate_init=0.001, learning_rate_end=1e-6, learning_rate_scheduler_type=LearningRateScheduler.NONE, momentum=0.9, weight_decay=0., save_file_name_prefix=None, save_every_sec=600, pre_generate_layers=False, distributed_backend=None, distributed_params_sync_every_iter=10000, distributed_num_buckets=None, dataloader_offline_num_workers=0, stop_with_bad_loss=True, log_file_name=None, lstm_dim=512, lstm_depth=1, proposal_mixture_components=10, num_workers = 1):
        if dataset_dir is None:
            dataset = OnlineDataset(model=self, inference_engine = inference_engine, inference_network=self._inference_network, importance_sample_size = importance_sample_size, ess_target = ess_target, prior_inflation=prior_inflation, num_workers=num_workers)
        else:
            dataset = OfflineDataset(dataset_dir=dataset_dir)

        if dataset_valid_dir is None:
            dataset_valid = None
        else:
            dataset_valid = OfflineDataset(dataset_dir=dataset_valid_dir)

        if self._inference_network is None:
            print('Creating new inference network...')
            if inference_network == InferenceNetwork.FEEDFORWARD:
                self._inference_network = InferenceNetworkFeedForward(model=self, observe_embeddings=observe_embeddings, proposal_mixture_components=proposal_mixture_components)
            elif inference_network == InferenceNetwork.LSTM:
                self._inference_network = InferenceNetworkLSTM(model=self, observe_embeddings=observe_embeddings, lstm_dim=lstm_dim, lstm_depth=lstm_depth, proposal_mixture_components=proposal_mixture_components)
            else:
                raise ValueError('Unknown inference_network: {}'.format(inference_network))
            if pre_generate_layers:
                if dataset_valid_dir is not None:
                    self._inference_network._pre_generate_layers(dataset_valid, save_file_name_prefix=save_file_name_prefix)
                if dataset_dir is not None:
                    self._inference_network._pre_generate_layers(dataset, save_file_name_prefix=save_file_name_prefix)
        else:
            print('Continuing to train existing inference network...')
            print('Total number of parameters: {:,}'.format(self._inference_network._history_num_params[-1]))

        self._inference_network.to(device=util._device)
        self._inference_network.optimize(num_traces=num_traces, dataset=dataset, dataset_valid=dataset_valid, num_traces_end=num_traces_end, batch_size=batch_size, valid_every=valid_every, optimizer_type=optimizer_type, learning_rate_init=learning_rate_init, learning_rate_end=learning_rate_end, learning_rate_scheduler_type=learning_rate_scheduler_type, momentum=momentum, weight_decay=weight_decay, save_file_name_prefix=save_file_name_prefix, save_every_sec=save_every_sec, distributed_backend=distributed_backend, distributed_params_sync_every_iter=distributed_params_sync_every_iter, distributed_num_buckets=distributed_num_buckets, dataloader_offline_num_workers=dataloader_offline_num_workers, stop_with_bad_loss=stop_with_bad_loss, log_file_name=log_file_name)

    def save_inference_network(self, file_name):
        if self._inference_network is None:
            raise RuntimeError('The model has no trained inference network.')
        self._inference_network._save(file_name)

    def load_inference_network(self, file_name):
        self._inference_network = InferenceNetworkBase._load(file_name)
        # The following is due to a temporary hack related with https://github.com/pytorch/pytorch/issues/9981 and can be deprecated by using dill as pickler with torch > 0.4.1
        self._inference_network._model = self

    def save_dataset(self, dataset_dir, num_traces, num_traces_per_file, prior_inflation=PriorInflation.DISABLED, *args, **kwargs):
        if not os.path.exists(dataset_dir):
            print('Directory does not exist, creating: {}'.format(dataset_dir))
            os.makedirs(dataset_dir)
        dataset = OnlineDataset(self, None, prior_inflation=prior_inflation)
        dataset.save_dataset(dataset_dir=dataset_dir, num_traces=num_traces, num_traces_per_file=num_traces_per_file, *args, **kwargs)

    def filter(self, filter, filter_timeout=1e6):
        return ConstrainedModel(self, filter=filter, filter_timeout=filter_timeout)


class RemoteModel(Model):
    def __init__(self, server_address='tcp://127.0.0.1:5555', before_forward_func=None, after_forward_func=None, *args, **kwargs):
        self._server_address = server_address
        self._model_server = None
        self._before_forward_func = before_forward_func  # Optional mthod to run before each forward call of the remote model (simulator)
        self._after_forward_func = after_forward_func  # Optional method to run after each forward call of the remote model (simulator)
        super().__init__(*args, **kwargs)

    def close(self):
        if self._model_server is not None:
            self._model_server.close()

    def forward(self):
        if self._model_server is None:
            self._model_server = ModelServer(self._server_address)
            self.name = '{} running on {}'.format(self._model_server.model_name, self._model_server.system_name)

        if self._before_forward_func is not None:
            self._before_forward_func()
        ret = self._model_server.forward()  # Calls the forward run of the remove model (simulator)
        if self._after_forward_func is not None:
            self._after_forward_func()
        return ret


class ConstrainedModel(Model):
    def __init__(self, base_model, filter, filter_timeout=1e6):
        self._base_model = base_model
        self._filter = filter
        self._filter_timeout = int(filter_timeout)
        # self.name = self._base_model.name
        # self._inference_network = self._base_model._inference_network
        # self._address_dictionary = self._base_model._address_dictionary

    def _trace_generator(self, *args, **kwargs):
        i = 0
        while True:
            i += 1
            if i > self._filter_timeout:
                raise RuntimeError('ConstrainedModel could not sample a trace satisfying the filter. Timeout ({}) reached.'.format(self._filter_timeout))
            trace = next(self._base_model._trace_generator(*args, **kwargs))
            if self._filter(trace):
                yield trace
            else:
                continue
