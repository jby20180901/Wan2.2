from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


class SimpleMirrorAPIBridge:
    def __init__(self, render_dir: str, pattern: str = "frame_{index:04d}.png"):
        self.render_dir = Path(render_dir)
        self.pattern = pattern

    def __call__(
        self,
        pose: np.ndarray,
        intrinsics: Optional[np.ndarray],
        frame_idx: int,
        height: int,
        width: int,
        mode: str = "rgb",
    ) -> np.ndarray:
        path = self.render_dir / self.pattern.format(index=frame_idx)
        if not path.exists():
            candidates = sorted(self.render_dir.glob(f"*{frame_idx:04d}*"))
            if not candidates:
                raise FileNotFoundError(f"frame not found for index {frame_idx} in {self.render_dir}")
            path = candidates[0]

        image = Image.open(path).convert("RGB" if mode == "rgb" else "L")
        image = image.resize((width, height), Image.BICUBIC)
        arr = np.asarray(image, dtype=np.float32)
        return arr


_default_bridge = None


def render_pose(
    pose: np.ndarray,
    intrinsics: Optional[np.ndarray],
    frame_idx: int,
    height: int,
    width: int,
    mode: str = "rgb",
):
    global _default_bridge
    if _default_bridge is None:
        raise RuntimeError("Initialize _default_bridge first, or replace render_pose with your HunyuanWorld-Mirror API call.")
    return _default_bridge(
        pose=pose,
        intrinsics=intrinsics,
        frame_idx=frame_idx,
        height=height,
        width=width,
        mode=mode,
    )
