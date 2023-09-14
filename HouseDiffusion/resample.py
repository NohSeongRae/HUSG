from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed as dist

def create_named_schedule_sampler(name, diffusion):
    pass

class ScheduleSampler(ABC):
    @abstractmethod
    def weights(self):
        pass

    def sample(self, batch_size, device):
        pass

class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        pass
    def weights(self):
        pass

class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        pass

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        pass

class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        pass
    def weights(self):
        pass
    def update_with_all_losses(self, ts, losses):
        pass
    def _warmed_up(self):
        pass
