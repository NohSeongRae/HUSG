"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""
import numpy as np

import torch


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the Kullback-Leibler (KL) divergence between two Gaussian distribution

    The KL divergence quantifies how one probability distribution diverges from a
    second, expected probability distribution.
    :param mean1:Mean of the first Gaussian distribution
    :param logvar1:Log variance of the first Gaussian distribution.
    :param mean2:Mean of the second Gaussian distribution
    :param logvar2:Log variance of the second Gaussian distribution.
    :return:The computed KL divergence.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
    Approximate the cumulative distribution function (CDF) of the standard normal distribution
    This function provides a fast approximation which can be useful in situations
    where performance is a concern
    :param x:Input tensor for which the CDF is computed
    :return:The approximated CDF of the standard normal for input x
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a discretized Gaussian distribution

    This is useful when the target data are quantized (e.g. images) and
    we want to model them with a Gaussian for generation
    :param x: Target images, assumed to be uint8 values rescaed to [-1, 1]
    :param means: Mean tensor of the Gaussian distribution
    :param log_scales: Log standard deviation tensor of the Gaussian distribution
    :return: A tensor of log probabilities (in nats) of the same shape as x.
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

