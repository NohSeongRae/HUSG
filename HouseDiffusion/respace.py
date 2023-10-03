import numpy as np
import torch

from .gaussian_diffusion import GaussianDiffusion

def space_timesteps(num_timesteps, section_counts):
    pass

class SpaceDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process
    use_timestep: a collection (sequence or set) of timesteps from the original  diffusion process to retain
    kwargs: the kwargs to create the base diffusion process
    """
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps=set(use_timesteps)
        self.timestep_map=[]
        self.original_num_steps=len(kwargs["betas"])

        base_diffusion=GaussianDiffusion(**kwargs)
        last_alpha_cumprod=1.0
        new_betas=[]
        for i, alpha_cumprod in enumerate(base_diffusion.alpha_cumprod)
    def p_mean_variance(self, model, *args, **kwargs):
        pass
    def training_losses(self, model, *args, **kwargs):
        pass
    def condition_mean(selfself, cond_fn, *args, **kwargs):
        pass
    def condition_score(self, cond_fn, *args, **kwargs):
        pass
    def _wrap_model(self, model):
        pass
    def _scale_timesteps(self, t):
        pass

class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        pass
    def __call__(self, x,ts, **kwargs):
        pass
