import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from exc
    return plt


def parse_step_dirs(steps_dir: Path):
    pattern = re.compile(r"^step_(\d+)_t(\d+)$")
    parsed = []
    for item in steps_dir.iterdir():
        if not item.is_dir():
            continue
        match = pattern.match(item.name)
        if not match:
            continue
        step_idx = int(match.group(1))
        timestep = int(match.group(2))
        parsed.append((step_idx, timestep, item))
    parsed.sort(key=lambda x: x[0])
    return parsed


def load_gray_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img.astype(np.float32)


def get_frequency_components(img: np.ndarray, radius: int):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    safe_radius = max(1, min(radius, min(crow, ccol) - 1))

    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), safe_radius, 1, thickness=-1)

    fshift_low = dft_shift * mask
    fshift_high = dft_shift * (1 - mask)

    img_low = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_low))).astype(np.float32)
    img_high = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_high))).astype(np.float32)
    return img_low, img_high


def mse(a: np.ndarray, b: np.ndarray):
    return float(np.mean((a - b) ** 2))


def normalize_convergence(data):
    arr = np.asarray(data, dtype=np.float64)
    d_min = np.min(arr)
    d_max = np.max(arr)
    if d_max == d_min:
        return np.ones_like(arr)
    return 1.0 - (arr - d_min) / (d_max - d_min)


def save_frame_csv(csv_path: Path, step_indices, timesteps, low_mse, high_mse, low_conv, high_conv):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step_index",
            "timestep",
            "low_mse",
            "high_mse",
            "low_convergence",
            "high_convergence",
        ])
        for i in range(len(step_indices)):
            writer.writerow([
                step_indices[i],
                timesteps[i],
                low_mse[i],
                high_mse[i],
                low_conv[i],
                high_conv[i],
            ])


def plot_frame_curve(plot_path: Path, title: str, step_indices, low_conv, high_conv):
    plt = _import_matplotlib()
    plt.figure(figsize=(10, 6))
    plt.plot(step_indices, low_conv, label="Low Frequency", color="blue", linewidth=2.2)
    plt.plot(
        step_indices,
        high_conv,
        label="High Frequency",
        color="orange",
        linewidth=2.2,
        linestyle="--",
    )
    plt.title(title)
    plt.xlabel("Diffusion Step Index")
    plt.ylabel("Convergence to Final (0-1)")
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze low/high-frequency convergence across diffusion steps for each frame.")
    parser.add_argument("--steps_dir", type=str, required=True, help="Directory containing step_xxx_txxxx folders with frame PNGs.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for charts and CSV files.")
    parser.add_argument("--radius", type=int, default=40, help="Low-frequency FFT radius.")
    parser.add_argument("--frame_pattern", type=str, default="frame_*.png", help="Frame file glob pattern inside each step folder.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap on number of frames to analyze.")
    args = parser.parse_args()

    steps_dir = Path(args.steps_dir)
    if not steps_dir.exists():
        raise FileNotFoundError(f"steps_dir does not exist: {steps_dir}")

    parsed_steps = parse_step_dirs(steps_dir)
    if len(parsed_steps) < 2:
        raise RuntimeError("Need at least 2 step folders to analyze convergence.")

    output_dir = Path(args.output_dir) if args.output_dir else steps_dir / "frequency_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_plot_dir = output_dir / "frame_plots"
    frame_csv_dir = output_dir / "frame_csv"
    frame_plot_dir.mkdir(parents=True, exist_ok=True)
    frame_csv_dir.mkdir(parents=True, exist_ok=True)

    final_step_dir = parsed_steps[-1][2]
    frame_paths = sorted(final_step_dir.glob(args.frame_pattern))
    if not frame_paths:
        raise RuntimeError(f"No frames found in final step folder: {final_step_dir}")

    frame_names = [p.name for p in frame_paths]
    if args.max_frames is not None and args.max_frames > 0:
        frame_names = frame_names[: args.max_frames]

    summary_low = []
    summary_high = []
    summary_steps = [s[0] for s in parsed_steps]

    for frame_name in tqdm(frame_names, desc="Analyzing frames"):
        valid_step_indices = []
        valid_timesteps = []
        images = []

        for step_idx, timestep, step_dir in parsed_steps:
            frame_path = step_dir / frame_name
            if not frame_path.exists():
                continue
            img = load_gray_image(frame_path)
            images.append(img)
            valid_step_indices.append(step_idx)
            valid_timesteps.append(timestep)

        if len(images) < 2:
            continue

        final_img = images[-1]
        low_final, high_final = get_frequency_components(final_img, args.radius)

        low_mse = []
        high_mse = []
        for img in images:
            if img.shape != final_img.shape:
                img = cv2.resize(img, (final_img.shape[1], final_img.shape[0]))
            low_img, high_img = get_frequency_components(img, args.radius)
            low_mse.append(mse(low_img, low_final))
            high_mse.append(mse(high_img, high_final))

        low_conv = normalize_convergence(low_mse)
        high_conv = normalize_convergence(high_mse)

        base = Path(frame_name).stem
        save_frame_csv(
            frame_csv_dir / f"{base}_convergence.csv",
            valid_step_indices,
            valid_timesteps,
            low_mse,
            high_mse,
            low_conv,
            high_conv,
        )
        plot_frame_curve(
            frame_plot_dir / f"{base}_convergence.png",
            title=f"Frequency Convergence - {frame_name}",
            step_indices=valid_step_indices,
            low_conv=low_conv,
            high_conv=high_conv,
        )

        if len(valid_step_indices) == len(summary_steps):
            summary_low.append(low_conv)
            summary_high.append(high_conv)

    if summary_low and summary_high:
        mean_low = np.mean(np.stack(summary_low, axis=0), axis=0)
        mean_high = np.mean(np.stack(summary_high, axis=0), axis=0)
        plot_frame_curve(
            output_dir / "mean_convergence_all_frames.png",
            title="Mean Frequency Convergence Across Frames",
            step_indices=summary_steps,
            low_conv=mean_low,
            high_conv=mean_high,
        )
        save_frame_csv(
            output_dir / "mean_convergence_all_frames.csv",
            summary_steps,
            [s[1] for s in parsed_steps],
            [0.0] * len(summary_steps),
            [0.0] * len(summary_steps),
            mean_low,
            mean_high,
        )

    print(f"Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
