import gc
import math
import random
import sys
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from ..image2video import WanI2V
from ..utils.diffusion_utils import IntermediateResultSaver
from ..utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .world_memory_guidance import (
    FlowMatchingFrequencyInjector,
    MirrorMemoryRetriever,
    PoseSequence,
)


class WanI2VWorldMemory(WanI2V):
    def generate(
        self,
        input_prompt,
        img,
        max_area=720 * 1280,
        frame_num=81,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        save_intermediate_dir=None,
        save_latents=True,
        save_decoded=False,
        memory_enable=False,
        memory_pose_file=None,
        memory_intrinsics_file=None,
        memory_api_target=None,
        memory_render_dir=None,
        memory_render_pattern="frame_{index:04d}.png",
        memory_render_mode="rgb",
        memory_fft_radius=10,
        memory_early_step_end=15,
        memory_mid_step_end=30,
        memory_lambda_early=0.7,
        memory_lambda_mid=0.2,
    ):
        guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
        )
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2]
        )
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                torch.zeros(3, F - 1, h, w)
            ], dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        memory_reference_latent = None
        memory_injector = None
        if memory_enable:
            if memory_pose_file is None:
                raise ValueError("memory_pose_file must be provided when memory_enable is True")
            pose_seq = PoseSequence.load(memory_pose_file, memory_intrinsics_file)
            retriever = MirrorMemoryRetriever(
                api_target=memory_api_target,
                render_dir=memory_render_dir,
                render_pattern=memory_render_pattern,
                mode=memory_render_mode,
            )
            memory_injector = FlowMatchingFrequencyInjector(
                vae=self.vae,
                num_train_timesteps=self.num_train_timesteps,
                radius=memory_fft_radius,
                early_step_end=memory_early_step_end,
                mid_step_end=memory_mid_step_end,
                lambda_early=memory_lambda_early,
                lambda_mid=memory_lambda_mid,
            )
            memory_reference_latent = memory_injector.prepare_reference_latent(
                retriever=retriever,
                poses=pose_seq.poses,
                intrinsics=pose_seq.intrinsics,
                frame_num=frame_num,
                height=h,
                width=w,
                device=self.device,
            )

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync', noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync', noop_no_sync)

        with (
            torch.amp.autocast('cuda', dtype=self.param_dtype),
            torch.no_grad(),
            no_sync_low_noise(),
            no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            result_saver = None
            if save_intermediate_dir is not None and self.rank == 0:
                result_saver = IntermediateResultSaver(
                    save_dir=save_intermediate_dir,
                    save_latents=save_latents,
                    save_decoded=save_decoded,
                    vae=self.vae if save_decoded else None
                )

            latent = noise

            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            for step_idx, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)

                model = self._prepare_model_for_timestep(t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]

                noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                if memory_injector is not None and memory_reference_latent is not None:
                    latent, _ = memory_injector.inject(
                        z_curr=latent,
                        z_ref=memory_reference_latent,
                        step_idx=step_idx,
                        timestep_value=float(t.item()),
                    )

                if result_saver is not None:
                    result_saver.save_step_results(
                        latents=latent,
                        step=t.item(),
                        step_idx=step_idx,
                        frame_num=frame_num,
                        vae_stride=self.vae_stride,
                    )

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if result_saver is not None and self.rank == 0:
            result_saver.save_final_result(videos[0], "final_video.pt")
            result_saver.get_summary()

        return videos[0] if self.rank == 0 else None
