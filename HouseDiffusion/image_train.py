import argparse

import dist_util
import logger
from rplanhg_datasets import load_rplanhg_data
from resample import create_named_schedule_sampler
from script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict, add_dict_to_argparser, \
    update_arg_parser
from trainutil import TrainLoop


def main():
    args = create_argparser().parse_args()
    update_arg_parser(args)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler=create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.dataset == 'rplan':
        data = load_rplanhg_data(
            batch_size=args.batch_size,
            analog_bit=args.analog_bit,
            target_set=args.target_set,
            set_name=args.set_name,
        )
    else:
        print('dataset not exist!')
        assert False

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        analog_bit=args.analog_bit,
    ).run_loop()


def create_argparser():
    defaults = dict(
        dataset='',
        schedule_sampler="uniform",  # "loss-second-moment", "uniform"
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_step=0,
        batch_size=1,
        microbatch=-1,  # disable microbatch
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    parser = argparse.ArgumentParser()
    defaults.update(model_and_diffusion_defaults())
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
