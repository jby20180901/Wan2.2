"""
Example script demonstrating how to save intermediate diffusion results during video generation.

This script shows how to use the new save_intermediate_dir feature in Wan2.2 to save
intermediate latent codes and optionally decoded frames at each diffusion step.
"""

import torch
from pathlib import Path
from wan import WanT2V
from wan.configs import wan_t2v_A14B_config

def example_t2v_with_intermediate_results():
    """
    Example: Generate video with T2V and save intermediate results
    """
    # Initialize model
    model = WanT2V(
        config=wan_t2v_A14B_config,
        checkpoint_dir="/path/to/checkpoints",
        device_id=0,
    )
    
    # Generate video with intermediate result saving
    # Directory structure will be:
    # outputs/intermediate_results/
    #   ├── step_000_t0999/
    #   │   ├── latents.pt (raw latent codes)
    #   │   └── frame_000.png, frame_001.png, ... (decoded frames, if save_decoded=True)
    #   ├── step_001_t0998/
    #   │   ├── latents.pt
    #   │   └── frame_000.png, frame_001.png, ...
    #   ├── ...
    #   └── final_video.pt (final decoded video)
    
    output = model.generate(
        input_prompt="A beautiful sunset over the ocean with waves crashing",
        size=(1280, 720),
        frame_num=81,
        sampling_steps=30,
        guide_scale=7.5,
        
        # New parameters for saving intermediate results
        save_intermediate_dir="./outputs/intermediate_results",
        save_latents=True,      # Save latent codes at each step (recommended)
        save_decoded=False,     # Set to True to also save decoded RGB images (slower)
    )
    
    # output shape: [3, 81, 720, 1280]
    return output


def example_i2v_with_intermediate_results():
    """
    Example: Generate video with I2V and save intermediate results
    """
    from PIL import Image
    from wan import WanI2V
    from wan.configs import wan_i2v_A14B_config
    
    # Initialize model
    model = WanI2V(
        config=wan_i2v_A14B_config,
        checkpoint_dir="/path/to/checkpoints",
        device_id=0,
    )
    
    # Load input image
    input_image = Image.open("path/to/image.jpg")
    
    # Generate video with intermediate saving
    output = model.generate(
        input_prompt="The person walks towards the camera",
        img=input_image,
        max_area=720 * 1280,
        frame_num=81,
        sampling_steps=40,
        guide_scale=5.0,
        
        # Save intermediate results
        save_intermediate_dir="./outputs/i2v_intermediate",
        save_latents=True,
        save_decoded=False,  # Warning: decoding at each step is memory-intensive!
    )
    
    return output


def load_and_analyze_intermediate_results(results_dir):
    """
    Example: Load and analyze intermediate results from a generation run.
    
    This shows how to:
    1. Load latent codes from each diffusion step
    2. Analyze how the latents evolve during generation
    3. Decode specific steps manually
    """
    import torch
    from pathlib import Path
    
    results_path = Path(results_dir)
    
    # Find all step directories
    step_dirs = sorted(results_path.glob("step_*"), key=lambda x: int(x.name.split('_')[1]))
    
    print(f"Found {len(step_dirs)} diffusion steps")
    print(f"Total steps: {len(step_dirs)}")
    print()
    
    # Analyze evolution
    for i, step_dir in enumerate(step_dirs[:5]):  # Show first 5 steps
        step_name = step_dir.name
        latents_path = step_dir / "latents.pt"
        
        if latents_path.exists():
            latents = torch.load(latents_path)
            print(f"{step_name}")
            print(f"  Latent shape: {latents.shape}")
            print(f"  Min: {latents.min():.4f}, Max: {latents.max():.4f}")
            print(f"  Mean: {latents.mean():.4f}, Std: {latents.std():.4f}")
            print()


def naming_convention_explanation():
    """
    Explanation of the directory and file naming convention for intermediate results.
    
    Directory Structure:
    ---------------------
    save_intermediate_dir/
    ├── step_000_t0999/          # First diffusion step, timestep 999
    │   ├── latents.pt           # Latent codes (shape: [C, T, H, W])
    │   ├── frame_000.png        # Decoded frame 0 (if save_decoded=True)
    │   ├── frame_001.png
    │   └── ... (81 frames total)
    │
    ├── step_001_t0998/          # Second step, timestep 998
    │   ├── latents.pt
    │   └── frame_*.png
    │
    ├── step_002_t0997/          # And so on...
    │   └── ...
    │
    └── final_video.pt           # Final decoded video
    
    Naming Details:
    ---------------
    - step_XXX: XXX is the step index (0, 1, 2, ..., sampling_steps-1)
    - t0000: the current diffusion timestep (from num_train_timesteps=1000 down to 0)
    - frame_YYY: YYY is the frame index within the video (0, 1, 2, ..., frame_num-1)
    
    Example for 50 sampling steps with 81 frames:
    - step_000_t0999: First reverse diffusion step, starting from heavy noise
    - step_001_t0998: Second step, slight denoising
    - ...
    - step_049_t0000: Final step, fully denoised output
    
    Each directory contains latent codes and optionally decoded frames for every frame
    in the video at that diffusion step.
    
    File Sizes:
    -----------
    - latents.pt: ~100 MB (for typical 81-frame, 720×1280 resolution)
    - Each frame PNG: ~1-5 MB (depends on compression)
    - Full step directory (with frames): ~100-500 MB per step
    
    Storage Estimation:
    -------------------
    For 50 sampling steps with save_latents=True, save_decoded=False:
    Total: ~50 × 100 MB = 5 GB
    
    For 50 sampling steps with both saved:
    Total: ~50 × 300 MB = 15 GB (very large!)
    
    Recommendations:
    ----------------
    1. Start with save_latents=True only (smaller disk space)
    2. Use save_decoded=True only if you need to visualize specific steps
    3. For analysis, it's better to load latents and decode manually as needed
    4. Consider using external storage for multi-step generation runs
    """
    print(__doc__)


if __name__ == "__main__":
    # Example usage
    print("Wan2.2 Intermediate Results Saving - Usage Examples")
    print("=" * 60)
    print()
    
    # Uncomment to run examples:
    # example_t2v_with_intermediate_results()
    # example_i2v_with_intermediate_results()
    # load_and_analyze_intermediate_results("./outputs/intermediate_results")
    
    naming_convention_explanation()
