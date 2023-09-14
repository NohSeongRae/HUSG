import enum
import math

import numpy as np
import torch

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from tqdm.auto import tqdm

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
     pass

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    pass

class ModelMeanType(enum.Enum):
    pass

class ModelVarType(enum.Enum):
    pass

class LossType(enum.Enum):
    pass

class GaussianDiffusion:
    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type, rescale_timestep=False):
        pass
    def q_mean_variance(self, x_start, t):
        pass

    def q_sample(self, x_start, t, noise=None):
        pass

    def q_posterior_mean_variance(self, x_start, x_t, t):
        pass

    def p_mean_variance(self, mode, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, analog_bit=None):
        def process_xstart(x):
            pass
    def _predict_xstart_from_eps(self, x_t, t, eps):
        pass

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        pass

    def _scale_timesteps(self, t):
        pass

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        pass

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        pass

    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, analog_bit=None):
        pass

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, analog_bit=None):
        pass

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, analog_bit=None):
        pass

    def ddim_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0.0):
        pass

    def ddim_reverse_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0):
        pass

    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0):
        pass

    def ddim_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0 ):
        pass

    def _vb_terms_pdf(self, model, x_start, x_t, t, padding_mask, clip_denoised=True, model_kwargs=None):
        pass

    def training_losses(self, model, x_start, t, model_kwargs, analog_bit, noise=None):
        def dec2bin(xinp, bits):
            pass

    def _prior_bpd(self, x_start):
        pass

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        pass

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    pass


