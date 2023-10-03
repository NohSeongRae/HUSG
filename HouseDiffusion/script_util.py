import argparse
import inspect

import gaussian_diffusion as gd
from respace import SpaceDiffusion, space_timesteps
from hd_transformer import TransformerModel


def diffusion_defaults():
    pass


def update_arg_parser(args):
    args.num_channels = 512
    num_coords = 16 if args.analog_bit else 2  # what is analog_bit?
    if args.dataset == 'rplan':
        args.input_channels = num_coords + (2 * 8 if not args.analog_bit else 0)
        args.condition_channel = 89  # why 89?
        args.out_channels = num_coords * 1
        args.use_unet = False

    elif args.dataset == 'st3d':
        args.input_channels = num_coords + (2 * 8 if not args.analog_bit else 0)
        args.condition_channels = 89
        args.out_channels = num_coords + 1
        args.use_unet = False
    elif args.dataset == 'zind':
        args.input_channels = num_coords + 2 * 8
        args.condition_channels = 89
        args.out_channels = num_coords * 1
        args.use_unet = False
    elif args.dataset == 'layout':
        args.use_unet = True
    elif args.dataset == 'outdoor':
        args.use_unet = True
    else:
        assert False, "Dataset Not Found"


def model_and_diffusion_defaults():
    pass


def create_model_and_diffusion(input_channels, condition_channels, num_channels, out_channels, dataset, use_checkpoint,
                               use_unet, learn_sigma, diffusoin_steps, noise_schedule,
                               timestep_respacing, use_kl, predict_xstart, rescale_timesteps, rescale_learned_sigmas,
                               analog_bit, target_set, set_name):
    model = TransformerModel(input_channels, condition_channels, num_channels, out_channels, dataset, use_checkpoint,
                             use_unet, analog_bit)
    diffusion = create_gaussian_diffusion(steps=diffusoin_steps, learn_sigma=learn_sigma, noise_schedule=noise_schedule,
                                          use_kl=use_kl, predict_xstart=predict_xstart,
                                          rescale_learned_sigma=rescale_learned_sigmas,
                                          timestep_respacing=timestep_respacing)
    return model, diffusion


def create_gaussian_diffusion(*, steps=1000, learn_sigma=False, sigma_small=False, noise_schedule="linear",
                              use_kl=False, predict_xstart=False,
                              rescales_timesteps=False, rescale_learned_sigmas=False, timestep_respacing="", ):
    betas=gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type=gd.LossType.RESCALED_MSE
    else:
        loss_type=gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing=[steps]
    return SpaceDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL

            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE

        ),
        loss_type=loss_type,
        rescale_timestep=rescales_timesteps

    )




def add_dict_to_argparser(parser, default_dict):
    pass


def args_to_dict(args, keys):
    pass


def str2bool(v):
    pass
