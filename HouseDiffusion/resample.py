from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    Factory function to instantiate a ScheduleSampler based on the provided name.

    :param name: the name/type of the ScheduleSampler to instantiate
    :param diffusion: The diffusion process object to be passed to the sampler
    :return:
    - ScheduleSampler: An instance of a subclass of ScheduleSampler
    :raise:
    - NotImplementedError: If the provided name does not match any known sampler
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    Abstract base calss for samplers that provide a distribution over timesteps
    in a diffusion process to reduce the variance of the obejctive.

    Subclasses are expected to provide an implementation of the weightss method and can override the
    sample method to adjust the reweighting of the resampled terms.
    """

    @abstractmethod
    def weights(self):
        """
        Abstract method to obtain the weights for each timestep
        :return:
        - np.ndarray: A NumPy array of positive weights for each diffusion step.
        """

    def sample(self, batch_size, device):
        """
        Sample timesteps for a batch based on the distribution provided by weights
        :param batch_size: (int) The number of timesteps to sample
        :param device: The torch device to which the resulting tensors should be moved

        Following code is implementation of uniform sample
        :return:
        - tuple: (timesteps, weights) where:
            - timestpes (torch.Tensor): A tensor of sampled timestep indices.
            - weights (torch.Tensor): A tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    """
    A ScheduleSampler subclass providing uniform sampling across timestep.
    """

    def __init__(self, diffusion):
        """
        Initialize the UniformSampler instance.
        :param diffusion: The diffusion process object
        """
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        """
        Get the uniform weights for each timestep
        :return:
        - np.ndarray: A NumPy array of ones, providing equal weights to each timestep.
        """
        return self._weights


class LossAwareSampler(ScheduleSampler):
    """
    Abstract base class for schedule samplers that update sampling weights
    based on losses obtained from a model
    """

    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model and synchronize
        across distributed training ranks.

        :param local_ts: (torch.Tensor) Tensor of timesteps.
        :param local_losses: (torch.Tensor) Corresponding losses for each timesteps
        :return: None
        """
        batch_sizes = [
            torch.tensor([0], dtype=torch.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [torch.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [torch.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Abstract method to update reweighting using losses

        :param ts: a list of int timesteps
        :param losses: Corresponding losses for each timesteps
        """


class LossSecondMomentResampler(LossAwareSampler):
    """
    A LossAwareSampler subclass that adjusts sampling weights based on the second moment of the loss
    """
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        """
        Initialize the LossSecondMomentResampler instance.
        :param diffusion: The diffusion process object
        :param history_per_term: (int) Number of historical loss values to store per timestep
        :param uniform_prob: (float) A small probability value to ensure non-zero sampling chance for all timesteps
        """
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        """
        Compute sampling weights based on the second moment of the loss and uniform_prob
        :return:
        - np.ndarray: Sampling weights for each timesteps.
        """
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        """
        Update _loss_history and _loss_counts based on provided losses.

        :param ts: (list[int]) List of timesteps
        :param losses: (list[float]) Corresponding losses for each timestep.
        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        """
        Check if sufficient loss history has been cumulated for all timesteps
        :return:
        -  bool: True if sufficient loss history is available, False otherwise
        """
        return (self._loss_counts == self.history_per_term).all()
