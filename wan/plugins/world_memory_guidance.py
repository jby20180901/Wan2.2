import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class PoseSequence:
    poses: torch.Tensor
    intrinsics: Optional[torch.Tensor] = None

    @staticmethod
    def load(pose_file: str, intrinsics_file: Optional[str] = None) -> "PoseSequence":
        pose_path = Path(pose_file)
        if not pose_path.exists():
            raise FileNotFoundError(f"pose file not found: {pose_file}")

        poses = _load_pose_or_intrinsic_tensor(pose_path, expected_last=(4, 4))

        intrinsics = None
        if intrinsics_file is not None:
            intr_path = Path(intrinsics_file)
            if not intr_path.exists():
                raise FileNotFoundError(f"intrinsics file not found: {intrinsics_file}")
            intrinsics = _load_pose_or_intrinsic_tensor(intr_path, expected_last=(3, 3))

        return PoseSequence(poses=poses.float(), intrinsics=intrinsics.float() if intrinsics is not None else None)


def _load_pose_or_intrinsic_tensor(path: Path, expected_last: Tuple[int, int]) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        if "c2w" in data and expected_last == (4, 4):
            arr = data["c2w"]
        elif "K" in data and expected_last == (3, 3):
            arr = data["K"]
        else:
            first_key = list(data.keys())[0]
            arr = data[first_key]
    elif path.suffix.lower() == ".json":
        arr = np.array(json.loads(path.read_text(encoding="utf-8")), dtype=np.float32)
    else:
        raise ValueError(f"unsupported file suffix: {path.suffix}")

    if arr.ndim != 3 or tuple(arr.shape[-2:]) != expected_last:
        raise ValueError(f"expected shape [N,{expected_last[0]},{expected_last[1]}], got {arr.shape} from {path}")
    return torch.from_numpy(arr)


class MirrorMemoryRetriever:
    def __init__(
        self,
        api_target: Optional[str] = None,
        render_dir: Optional[str] = None,
        render_pattern: str = "frame_{index:04d}.png",
        mode: str = "rgb",
    ):
        self.mode = mode
        self.render_dir = Path(render_dir) if render_dir else None
        self.render_pattern = render_pattern
        self.api_fn = _load_api_function(api_target) if api_target else None

        if self.api_fn is None and self.render_dir is None:
            raise ValueError("either api_target or render_dir must be provided")

    def render_frame(
        self,
        frame_idx: int,
        pose: torch.Tensor,
        intrinsics: Optional[torch.Tensor],
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.api_fn is not None:
            output = self.api_fn(
                pose=pose.detach().cpu().numpy(),
                intrinsics=None if intrinsics is None else intrinsics.detach().cpu().numpy(),
                frame_idx=frame_idx,
                height=height,
                width=width,
                mode=self.mode,
            )
            image = _to_image_tensor(output, height=height, width=width, mode=self.mode, device=device)
            return image

        path = self._resolve_render_path(frame_idx)
        image = Image.open(path)
        if self.mode == "depth":
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        arr = np.array(image)
        return _to_image_tensor(arr, height=height, width=width, mode=self.mode, device=device)

    def _resolve_render_path(self, frame_idx: int) -> Path:
        if self.render_dir is None:
            raise RuntimeError("render_dir is not configured")

        direct = self.render_dir / self.render_pattern.format(index=frame_idx)
        if direct.exists():
            return direct

        candidates = sorted(self.render_dir.glob(f"*{frame_idx:04d}*"))
        if candidates:
            return candidates[0]

        raise FileNotFoundError(f"render frame not found for index {frame_idx} in {self.render_dir}")


def _load_api_function(api_target: str) -> Callable:
    if ":" not in api_target:
        raise ValueError("api_target must be module:function, e.g. third_party_api.renderer:render_pose")
    module_name, fn_name = api_target.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name, None)
    if fn is None:
        raise ValueError(f"function {fn_name} not found in module {module_name}")
    if not callable(fn):
        raise ValueError(f"{api_target} is not callable")
    return fn


def _to_image_tensor(output, height: int, width: int, mode: str, device: torch.device) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        tensor = output.detach().float()
    else:
        tensor = torch.from_numpy(np.asarray(output)).float()

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[0] in (1, 3):
            pass
        elif tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"cannot infer image channel axis from shape {tuple(tensor.shape)}")
    else:
        raise ValueError(f"unsupported render output shape: {tuple(tensor.shape)}")

    tensor = tensor.to(device)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    if tensor.shape[-2] != height or tensor.shape[-1] != width:
        tensor = F.interpolate(tensor.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)

    max_val = tensor.max().item() if tensor.numel() > 0 else 1.0
    if max_val > 1.5:
        tensor = tensor / 255.0

    tensor = tensor.clamp(0.0, 1.0)
    if mode == "depth":
        tensor = tensor.mean(dim=0, keepdim=True).repeat(3, 1, 1)

    return tensor.mul(2.0).sub(1.0)


class FlowMatchingFrequencyInjector:
    def __init__(
        self,
        vae,
        num_train_timesteps: int,
        radius: int = 10,
        early_step_end: int = 15,
        mid_step_end: int = 30,
        lambda_early: float = 0.7,
        lambda_mid: float = 0.2,
    ):
        self.vae = vae
        self.num_train_timesteps = num_train_timesteps
        self.radius = radius
        self.early_step_end = early_step_end
        self.mid_step_end = mid_step_end
        self.lambda_early = lambda_early
        self.lambda_mid = lambda_mid

    def prepare_reference_latent(
        self,
        retriever: MirrorMemoryRetriever,
        poses: torch.Tensor,
        intrinsics: Optional[torch.Tensor],
        frame_num: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        frame_num = min(frame_num, poses.shape[0])
        if frame_num <= 0:
            raise ValueError("frame_num must be > 0 for memory retrieval")

        frames = []
        for frame_idx in range(frame_num):
            pose = poses[frame_idx]
            intr = intrinsics[frame_idx] if intrinsics is not None and frame_idx < intrinsics.shape[0] else None
            frame = retriever.render_frame(
                frame_idx=frame_idx,
                pose=pose,
                intrinsics=intr,
                height=height,
                width=width,
                device=device,
            )
            frames.append(frame)

        video = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
        with torch.no_grad():
            latent = self.vae.encode([video])[0]
        return latent

    def inject(
        self,
        z_curr: torch.Tensor,
        z_ref: torch.Tensor,
        step_idx: int,
        timestep_value: float,
    ) -> Tuple[torch.Tensor, float]:
        lambda_step = self._guidance_lambda(step_idx)
        if lambda_step <= 0:
            return z_curr, 0.0

        if z_curr.ndim not in (4, 5):
            raise ValueError(f"expected latent ndim 4 or 5, got {z_curr.ndim}")

        z_curr_batched, was_4d = _ensure_batched(z_curr)
        z_ref_batched, _ = _ensure_batched(z_ref)

        z_ref_batched = _resize_5d_to_match(z_ref_batched, z_curr_batched)

        t_norm = float(np.clip(timestep_value / float(self.num_train_timesteps), 0.0, 1.0))
        eps = torch.randn(
            z_ref_batched.shape,
            device=z_ref_batched.device,
            dtype=z_ref_batched.dtype,
        )
        z_ref_noisy = (1.0 - t_norm) * z_ref_batched + t_norm * eps
        z_ref_aligned = _match_mean_std(z_ref_noisy, z_curr_batched)

        z_injected = _fft_low_high_mix(z_ref_aligned, z_curr_batched, self.radius)
        z_out = (1.0 - lambda_step) * z_curr_batched + lambda_step * z_injected

        if was_4d:
            z_out = z_out.squeeze(0)
        return z_out, lambda_step

    def _guidance_lambda(self, step_idx: int) -> float:
        if step_idx < self.early_step_end:
            return self.lambda_early
        if step_idx < self.mid_step_end:
            span = max(1, self.mid_step_end - self.early_step_end)
            ratio = float(step_idx - self.early_step_end) / float(span)
            return self.lambda_early + ratio * (self.lambda_mid - self.lambda_early)
        return 0.0


def _ensure_batched(z: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if z.ndim == 4:
        return z.unsqueeze(0), True
    return z, False


def _resize_5d_to_match(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if src.shape == ref.shape:
        return src
    return F.interpolate(src, size=ref.shape[-3:], mode="trilinear", align_corners=False)


def _match_mean_std(source: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    source_mean = source.mean(dim=(-3, -2, -1), keepdim=True)
    source_std = source.std(dim=(-3, -2, -1), keepdim=True).clamp_min(eps)
    target_mean = target.mean(dim=(-3, -2, -1), keepdim=True)
    target_std = target.std(dim=(-3, -2, -1), keepdim=True).clamp_min(eps)
    return (source - source_mean) / source_std * target_std + target_mean


def _fft_low_high_mix(z_low_source: torch.Tensor, z_high_source: torch.Tensor, radius: int) -> torch.Tensor:
    b, c, t, h, w = z_low_source.shape
    crow = h // 2
    ccol = w // 2
    safe_radius = int(max(1, min(radius, min(crow, ccol) - 1)))

    yy, xx = torch.meshgrid(
        torch.arange(h, device=z_low_source.device),
        torch.arange(w, device=z_low_source.device),
        indexing="ij",
    )
    mask_2d = ((yy - crow) ** 2 + (xx - ccol) ** 2 <= safe_radius * safe_radius).to(z_low_source.dtype)
    mask = mask_2d.view(1, 1, 1, h, w)

    low_fft = torch.fft.fftshift(torch.fft.fft2(z_low_source, dim=(-2, -1)), dim=(-2, -1))
    high_fft = torch.fft.fftshift(torch.fft.fft2(z_high_source, dim=(-2, -1)), dim=(-2, -1))

    merged_fft = low_fft * mask + high_fft * (1.0 - mask)
    merged = torch.fft.ifft2(torch.fft.ifftshift(merged_fft, dim=(-2, -1)), dim=(-2, -1)).real
    return merged.reshape(b, c, t, h, w)
