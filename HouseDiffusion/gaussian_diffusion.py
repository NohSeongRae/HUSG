"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
import enum
import math

import numpy as np
import torch

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from tqdm.auto import tqdm


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    get a pre-defined beta schedule for the given name

    the beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps
    Beta schedules may be added, but should not be removed or changed once they are
    committed to maintain backward compatibility
    """
    if schedule_name == "linear":
        # linear schedule from Ho et al, extended to work for any number of diffusion steps
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        print("COSINE")
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t) / 1.000 * math.pi / 2) ** 2
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1]

    :param num_diffusion_timesteps: the number of betas to produce
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                    produces the cumulative product of (1-beta) up to that part of the diffusion process
    :param max_beta: the maximum beta to use. use values lower than 1 to prevent singularities
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts
    """

    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The Learned_range option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier
    """
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss and KL when learning variance
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE with RESCALED_KL
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models

    Ported directly from here, and then adapted over time to further experiments
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep, starting at T and going to 1
    :param model_mean_type: a ModelMeanType determining what the model outputs
    :param model_var_type: a ModelVarType determining how variance is output
    :param loss_type: a LossType determining the loss function to use
    :param rescale_timesteps: if True, pass floating point timesteps into the
                            model so that they are always scaled like in the original paper (0 to 1000)
    """

    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type, rescale_timestep=False):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timestep

        # Use float64 for accuracy
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        # ensure betas are one-dimensional array and in the (0,1] range
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,) #ensure shape of alphas.cumprod_prev consistent with timestep

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t|x_0)
        :param x_start: the [ N x C  x ... ] tensor of noiseless inputs
        :param t: the number of diffusion steps (minus 1). In here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape
        """
        mean=(
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance=_extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps
        In other words, sample from q(x_t|x_0)
        :param x_start: the initial data batch
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise:if specified, the split-out normal noise
        :return:a noisy version of x_start
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape ==x_start.shape
        return(
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            *noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1}|x_t, x_0)
        :param x_start:the inital data batch
        :param x_t:t step data
        :param t:time step
        :return:posterior mean, variance, log_variance_clipped
        """
        assert x_start.shape ==x_t.shape
        posterior_mean=(
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert(
            posterior_mean.shape[0]
            ==posterior_variance.shape[0]
            ==posterior_log_variance_clipped.shape[0]
            ==x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, analog_bit=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the inital x, x_0
        :param mode: the model, which takes a signal and a batch of timesteps as input
        :param x: the [N x C x ...] tensor at time t
        :param t:a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal clamped into [-1, 1]
        :param denoised_fn: An optional function that processes the prediction of x_start befor its used
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model.
                            This can be used for conditioning
        :param analog_bit:
        :return:
        """
        if model_kwargs is None:
            model_kwargs={}

        B, C = x.shape[:2]
        assert t.shape ==(B,)
        xtalpha=_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape).permute([0,2,1])
        epsalpha=_extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape).permute([0,2,1])
        model_output_dec, model_output_bin=model(x, self._scale_timesteps(t), xtalpha=xtalpha, epsalpha=epsalpha, is_syn=True, **model_kwargs)
        model_output=model_output_dec

        if analog_bit:
            predict_discrete = 0
        else:
            predict_discrete=32
        if t[0] <predict_discrete:
            def bin2dec(b, bits):
                mask = 2** torch.arange(bits-1, -1, -1).to(b.device, b.dtype)
                return torch.sum(mask*b, -1)
            model_output_bin[model_output_bin>0]=1
            model_output_bin[model_output_bin<=0]=0
            model_output_bin=bin2dec(model_output_bin.round().int().permute([0,2,1]).reshape(model_output_bin.shape[0],
                            model_output_bin.shape[2], 2, 8), 8).permute([0,2,1])
            model_output_bin=((model_output_bin/256) -0.5)*2
            model_output=model_output_bin

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B,C *2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type==ModelVarType.LEARNED:
                model_long_variance=model_var_values
                model_variance=torch.exp(model_long_variance)
            else:
                min_log=_extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log=_extract_into_tensor(np.log(self.betas), t, x.shape)
                #The model_var_values is [-1, 1] for [min_var, max_var]
                frac=(model_var_values+1)/2
                model_log_variance=frac*max_log + (1-frac) * min_log
                model_variance=torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixed_large, we set the inital (log-) variance like so
                # to get a better decoder log-likelihood
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance=_extract_into_tensor(model_variance, t, x.shape)
            model_log_variance=_extract_into_tensor(model_log_variance, t, x.shape)
        def process_xstart(x):
            if denoised_fn is not None:
                x=denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if t[0] >= predict_discrete:
            if self.model_mean_type == ModelMeanType.PREVIOUS_X:
                pred_xstart=process_xstart(
                    self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
                )
                model_mean=model_output
            elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
                if self.model_mean_type ==ModelMeanType.START_X:
                    pred_xstart=process_xstart(model_output)
                else:
                    pred_xstart=process_xstart(
                        self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output )
                    )
                model_mean,_, _=self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
            else:
                raise NotImplementedError(self.model_mean_type)
        else:
            pred_xstart=process_xstart(model_output)
            model_mean,_, _=self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        assert(
            model_mean.shape==model_log_variance.shape==pred_xstart.shape==x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }





    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape ==eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) *x_t
             - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return ( # (xprev - coef2 *t_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t , x_t.shape) *xprev
            - _extract_into_tensor(self.posterior_mean_coef2/self.posterior_mean_coef1,t , x_t.shape)* x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        pass

    def _scale_timesteps(self, t):
        pass

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        pass

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        pass

    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None,
                 analog_bit=None):
        pass

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None,
                      model_kwargs=None, device=None, progress=False, analog_bit=None):
        pass

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None,
                                  model_kwargs=None, device=None, progress=False, analog_bit=None):
        pass

    def ddim_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0.0):
        pass

    def ddim_reverse_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0):
        pass

    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None,
                         model_kwargs=None, device=None, progress=False, eta=0.0):
        pass

    def ddim_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None,
                                     model_kwargs=None, device=None, progress=False, eta=0.0):
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
    """
    Extract values from a 1-D numpy array for a batch of indices
    :param arr:the 1-D numpy array
    :param timesteps: a tensor containing indices to extract from arr
    :param broadcast_shape: a shape defining the target shape of the output tensor
    :return: a tensor of shape [batch_size, 1, ...] shere the shape has K dims
    """
    res=torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape)<len(broadcast_shape):
        res=res[..., None]
    return res.expand(broadcast_shape)
