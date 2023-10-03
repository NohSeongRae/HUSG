import argparse

import dist_util
import logger
from rplanhg_datasets import load_rplanhg_data
from resample import create_named_schedule_sampler
from script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict, add_dict_to_argparser, update_arg_parser
from trainutil import TrainLoop

def main():
    args=create_argparser().parse_args()
    update_arg_parser(args)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

def create_argparser():
    defaults=dict(
        dataset='',
        schedule_sampler="uniform", #"loss-second-moment", "uniform"
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_step=0,
        batch_size=1,
        microbatch=-1, # disable microbatch
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    parser=argparse.ArgumentParser()
    defaults.update(model_and_diffusion_defaults())
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__=="__main__":
    main()