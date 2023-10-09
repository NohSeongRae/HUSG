import numpy as np
import torch

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Generate a set of diffusion steps based on specified spacing logic
    :param num_timesteps: (int) Total number of timesteps in the original diffusion process.
    :param section_counts: (Union[str, List[int]]) Defines how to space out or select timesteps.
        - If str and starts with "ddim", uses the DDIM striding.
        - If str of comma-separated numbers, converts to list to ints.
        - If list of ints, uses there as the count of timesteps per section
    :return:
        set: Selected timesteps based on the spacing logic
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_counts in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_counts:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_counts}"
            )
        if section_counts <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_counts - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_counts):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpaceDiffusion(GaussianDiffusion):
    """
    A diffusion process model allowing for using selective timesteps from an original diffusion process.

    Attributes:
        use_timesteps (set): A set of timesteps to be used from the original diffusion process
        timestep_map (list): Maps the modified timesteps to original timesteps.
        original_num_timesteps (int): Original number of steps in the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        """
        Initialize the SpaceDiffusion instance
        :param use_timesteps (iterable):The timesteps to be used from the origianl diffusion process
        :param kwargs: additional keyword arguments for initializing the GaussianDiffusion base class
        """
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alpha_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by wrapped model
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
