# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np

__all__ = ['IntermediateResultSaver']


class IntermediateResultSaver:
    """
    Save intermediate results during diffusion process.
    
    This class handles saving latent codes at each diffusion step for each frame,
    and can optionally decode them to RGB images.
    """
    
    def __init__(self, save_dir, save_latents=True, save_decoded=False, vae=None):
        """
        Args:
            save_dir (str): Directory to save intermediate results
            save_latents (bool): Whether to save raw latent codes as .pt files
            save_decoded (bool): Whether to save decoded RGB images
            vae (nn.Module): VAE decoder, required if save_decoded=True
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_latents = save_latents
        self.save_decoded = save_decoded
        self.vae = vae
        
        if save_decoded and vae is None:
            raise ValueError("VAE decoder is required when save_decoded=True")
    
    def save_step_results(self, latents, step, step_idx, frame_num, vae_stride, 
                         decoded_vae_input=None):
        """
        Save intermediate results for a specific diffusion step.
        
        Args:
            latents (torch.Tensor): Latent codes, shape [C, T, H, W]
            step (int): Current diffusion step (e.g., 999, 998, ...)
            step_idx (int): Step index in sampling schedule (0, 1, 2, ...)
            frame_num (int): Total number of frames
            vae_stride (tuple): VAE stride for each dimension (T, H, W)
            decoded_vae_input (torch.Tensor): Optional pre-decoded VAE output for efficiency
        """
        # Create step directory
        step_dir = self.save_dir / f"step_{step_idx:03d}_t{step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        
        # Save latent codes if requested
        if self.save_latents:
            latent_path = step_dir / "latents.pt"
            torch.save(latents.cpu(), latent_path)
        
        # Save decoded images if requested
        if self.save_decoded:
            self._save_decoded_frames(
                latents, step_dir, frame_num, vae_stride, decoded_vae_input
            )
    
    def _save_decoded_frames(self, latents, step_dir, frame_num, vae_stride, 
                            decoded_vae_input=None):
        """
        Decode and save individual frames as PNG images.
        
        Args:
            latents (torch.Tensor): Latent codes, shape [C, T, H, W]
            step_dir (Path): Directory to save frames
            frame_num (int): Total number of frames
            vae_stride (tuple): VAE stride for each dimension (T, H, W)
            decoded_vae_input (torch.Tensor): Optional pre-decoded VAE output
        """
        try:
            # If pre-decoded output is provided, use it
            if decoded_vae_input is not None:
                video = decoded_vae_input
            else:
                # Decode latents to video
                with torch.no_grad():
                    video = self.vae.decode([latents])[0]  # [3, T, H, W]
            
            # Save each frame
            video = video.cpu().numpy()
            video = np.transpose(video, (1, 2, 3, 0))  # [T, H, W, 3]
            
            # Normalize to 0-255 if needed
            if video.dtype == np.float32 or video.dtype == np.float64:
                video = np.clip(video * 255, 0, 255).astype(np.uint8)
            
            for frame_idx in range(frame_num):
                frame = video[frame_idx]
                frame_path = step_dir / f"frame_{frame_idx:03d}.png"
                Image.fromarray(frame).save(frame_path)
        
        except Exception as e:
            print(f"Warning: Failed to decode and save frames: {e}")
    
    def save_final_result(self, video, filename="final_video.pt"):
        """
        Save the final generated video.
        
        Args:
            video (torch.Tensor): Final video tensor
            filename (str): Filename for the final video
        """
        final_path = self.save_dir / filename
        torch.save(video.cpu(), final_path)
    
    def get_summary(self):
        """
        Print summary of saved intermediate results.
        """
        steps = [d for d in self.save_dir.iterdir() if d.is_dir() and d.name.startswith('step_')]
        steps.sort(key=lambda x: int(x.name.split('_')[1]))
        
        print(f"\n{'='*60}")
        print(f"Intermediate Results Summary")
        print(f"{'='*60}")
        print(f"Total steps saved: {len(steps)}")
        print(f"Save directory: {self.save_dir}")
        print(f"Latents saved: {self.save_latents}")
        print(f"Decoded frames saved: {self.save_decoded}")
        
        if steps:
            print(f"\nFirst step: {steps[0].name}")
            print(f"Last step: {steps[-1].name}")
            
            # Show structure of first step
            first_step_files = list(steps[0].iterdir())
            print(f"\nFiles in first step directory:")
            for f in first_step_files:
                if f.is_file():
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"  - {f.name} ({size_mb:.2f} MB)")
                elif f.is_dir():
                    frame_count = len(list(f.glob("*.png")))
                    print(f"  - {f.name}/ ({frame_count} frames)")
        print(f"{'='*60}\n")
