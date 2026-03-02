import argparse
import logging
import os
import random
import sys
from datetime import datetime

import torch
import torch.distributed as dist
from PIL import Image

from wan.configs import MAX_AREA_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.plugins.world_memory_i2v import WanI2VWorldMemory
from wan.utils.utils import save_video, str2bool


def _parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2 I2V with HunyuanWorld-Mirror external memory guidance")
    parser.add_argument("--task", type=str, default="i2v-A14B", choices=["i2v-A14B"])
    parser.add_argument("--size", type=str, default="1280*720")
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)
    parser.add_argument("--save_intermediate_dir", type=str, default=None)
    parser.add_argument("--save_latents", type=str2bool, default=True)
    parser.add_argument("--save_decoded", type=str2bool, default=False)

    parser.add_argument("--memory_enable", type=str2bool, default=True)
    parser.add_argument("--memory_pose_file", type=str, default=None)
    parser.add_argument("--memory_intrinsics_file", type=str, default=None)
    parser.add_argument("--memory_api_target", type=str, default=None)
    parser.add_argument("--memory_render_dir", type=str, default=None)
    parser.add_argument("--memory_render_pattern", type=str, default="frame_{index:04d}.png")
    parser.add_argument("--memory_render_mode", type=str, default="rgb", choices=["rgb", "depth"])
    parser.add_argument("--memory_fft_radius", type=int, default=10)
    parser.add_argument("--memory_early_step_end", type=int, default=15)
    parser.add_argument("--memory_mid_step_end", type=int, default=30)
    parser.add_argument("--memory_lambda_early", type=float, default=0.7)
    parser.add_argument("--memory_lambda_mid", type=float, default=0.2)

    args = parser.parse_args()
    cfg = WAN_CONFIGS[args.task]

    if args.size not in SUPPORTED_SIZES[args.task]:
        raise ValueError(
            f"unsupported size {args.size} for {args.task}, available: {SUPPORTED_SIZES[args.task]}"
        )

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    if args.base_seed < 0:
        args.base_seed = random.randint(0, sys.maxsize)

    if args.memory_enable and args.memory_pose_file is None:
        raise ValueError("--memory_pose_file is required when --memory_enable=true")
    if args.memory_enable and args.memory_api_target is None and args.memory_render_dir is None:
        raise ValueError("provide --memory_api_target or --memory_render_dir when --memory_enable=true")

    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def main():
    args = _parse_args()

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    else:
        if args.t5_fsdp or args.dit_fsdp:
            raise ValueError("t5_fsdp and dit_fsdp are not supported in non-distributed mode")
        if args.ulysses_size > 1:
            raise ValueError("ulysses_size > 1 is not supported in non-distributed mode")

    if args.ulysses_size > 1:
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1 and cfg.num_heads % args.ulysses_size != 0:
        raise ValueError(f"{cfg.num_heads=} cannot be evenly divided by {args.ulysses_size=}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Job args: {args}")

    img = Image.open(args.image).convert("RGB")

    pipeline = WanI2VWorldMemory(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    video = pipeline.generate(
        input_prompt=args.prompt,
        img=img,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
        save_intermediate_dir=args.save_intermediate_dir,
        save_latents=args.save_latents,
        save_decoded=args.save_decoded,
        memory_enable=args.memory_enable,
        memory_pose_file=args.memory_pose_file,
        memory_intrinsics_file=args.memory_intrinsics_file,
        memory_api_target=args.memory_api_target,
        memory_render_dir=args.memory_render_dir,
        memory_render_pattern=args.memory_render_pattern,
        memory_render_mode=args.memory_render_mode,
        memory_fft_radius=args.memory_fft_radius,
        memory_early_step_end=args.memory_early_step_end,
        memory_mid_step_end=args.memory_mid_step_end,
        memory_lambda_early=args.memory_lambda_early,
        memory_lambda_mid=args.memory_lambda_mid,
    )

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            args.save_file = f"memory_i2v_{args.size}_{formatted_prompt}_{formatted_time}.mp4"

        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        logging.info(f"Saved video to {args.save_file}")

    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
