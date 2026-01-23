# Wan2.2 中间结果保存功能文档

## 功能概述

在 Wan2.2 中，你现在可以在视频生成过程中保存**每一帧在每一个扩散步的中间生成结果**。这对于以下场景很有用：

- 📊 分析扩散过程如何逐步生成视频
- 🎬 可视化生成过程的演化
- 🔬 研究模型行为和学习动态
- 🎨 从特定扩散步提取高质量的中间结果
- 📈 调试和优化生成参数

## 快速开始

### 基本用法

在调用 `generate()` 方法时添加以下参数：

```python
from wan import WanT2V
from wan.configs import wan_t2v_A14B_config

model = WanT2V(
    config=wan_t2v_A14B_config,
    checkpoint_dir="./checkpoints",
    device_id=0,
)

video = model.generate(
    input_prompt="A beautiful sunset over the ocean",
    size=(1280, 720),
    frame_num=81,
    sampling_steps=30,
    guide_scale=7.5,
    
    # 保存中间结果的参数
    save_intermediate_dir="./outputs/intermediate_results",
    save_latents=True,      # 保存潜在代码（推荐）
    save_decoded=False,     # 保存解码后的 RGB 图像（可选，较慢）
)
```

## 参数说明

### `save_intermediate_dir` (str, 可选)
- **默认值**: `None` (不保存)
- **说明**: 保存中间结果的目录路径。如果设置，将在该目录创建目录结构来保存每一步的结果。

### `save_latents` (bool, 可选)
- **默认值**: `True`
- **说明**: 是否保存原始潜在代码（`.pt` 文件格式）
- **推荐**: 始终为 `True`，占用空间较小（~100 MB/step）

### `save_decoded` (bool, 可选)
- **默认值**: `False`
- **说明**: 是否在每一步解码并保存 RGB 图像（PNG 格式）
- **警告**: 启用此选项会显著增加：
  - 计算时间（需要在每步调用 VAE 解码）
  - 磁盘空间（~300-500 MB/step）
  - 内存使用（可能导致 OOM）

## 目录结构

生成的中间结果将按以下方式组织：

```
./outputs/intermediate_results/
├── step_000_t0999/          # 第 0 步，时间步 999
│   ├── latents.pt           # 潜在代码 [C, T, H, W]
│   ├── frame_000.png        # 第 0 帧（仅当 save_decoded=True）
│   ├── frame_001.png
│   └── frame_080.png        # 第 80 帧
│
├── step_001_t0998/          # 第 1 步，时间步 998
│   ├── latents.pt
│   └── frame_*.png
│
├── step_002_t0997/
│   └── ...
│
├── ... (继续 sampling_steps-1 步)
│
└── final_video.pt           # 最终生成的视频
```

### 命名约定说明

- **step_XXX**: 步数索引（0 到 sampling_steps-1）
- **t0YYY**: 当前扩散时间步（从 num_train_timesteps=1000 倒数到 0）
- **frame_ZZZ**: 视频内的帧索引（0 到 frame_num-1）

**例子**：对于 50 个采样步和 81 帧：
- `step_000_t0999/`: 开始步，高度噪声状态
- `step_001_t0998/`: 第二步，开始去噪
- `step_049_t0000/`: 最后一步，完全去噪

## 所有支持的模型

该功能已集成到以下生成类中：

| 类 | 文件 | 说明 |
|------|---------|------|
| `WanT2V` | `text2video.py` | 文本转视频 |
| `WanI2V` | `image2video.py` | 图像转视频 |
| `WanTI2V` | `textimage2video.py` | 文本+图像转视频 |
| `WanS2V` | `speech2video.py` | 语音转视频 |
| `WanAnimate` | `animate.py` | 角色动画 |

使用方式相同：
```python
# I2V 例子
video = model.generate(
    input_prompt="...",
    img=image,
    save_intermediate_dir="./outputs",
    save_latents=True,
    save_decoded=False,
)
```

## 存储空间估算

### 仅保存潜在代码 (`save_latents=True, save_decoded=False`)

| 采样步数 | 帧数 | 总大小 |
|---------|------|--------|
| 30 步 | 81 | ~3 GB |
| 50 步 | 81 | ~5 GB |
| 100 步 | 81 | ~10 GB |

### 保存潜在代码和解码图像 (`save_latents=True, save_decoded=True`)

| 采样步数 | 帧数 | 总大小 |
|---------|------|--------|
| 30 步 | 81 | ~9 GB |
| 50 步 | 81 | ~15 GB |
| 100 步 | 81 | ~30 GB |

## 性能影响

### 保存潜在代码
- **时间增加**: ~0-1% (保存很快)
- **内存增加**: 最小（异步保存）
- **推荐**: 始终启用

### 保存解码图像
- **时间增加**: 30-40% (需要 VAE 解码)
- **内存增加**: 显著（每步 ~1-2 GB）
- **建议**: 仅在需要可视化时启用

## 加载和使用中间结果

### 加载潜在代码

```python
import torch
from pathlib import Path

# 加载特定步骤的潜在代码
latents = torch.load("./outputs/intermediate_results/step_010_t0989/latents.pt")
print(latents.shape)  # [C, T, H, W]
```

### 手动解码特定步骤

```python
import torch
from wan import WanT2V
from wan.configs import wan_t2v_A14B_config

# 初始化模型和 VAE
model = WanT2V(config=wan_t2v_A14B_config, checkpoint_dir="./checkpoints")

# 加载潜在代码
latents = torch.load("./outputs/intermediate_results/step_010_t0989/latents.pt")

# 手动解码
with torch.no_grad():
    video = model.vae.decode([latents])  # [1, 3, T, H, W]
    
print(video[0].shape)  # [3, 81, 720, 1280]
```

### 分析扩散演化

```python
import torch
from pathlib import Path

results_dir = "./outputs/intermediate_results"
step_dirs = sorted(Path(results_dir).glob("step_*"))

for step_dir in step_dirs[:10]:  # 分析前 10 步
    latents = torch.load(step_dir / "latents.pt")
    
    print(f"{step_dir.name}:")
    print(f"  Min: {latents.min():.4f}, Max: {latents.max():.4f}")
    print(f"  Mean: {latents.mean():.4f}, Std: {latents.std():.4f}")
    print(f"  Norm: {latents.norm():.4f}")
```

## 常见问题

### Q1: 保存解码图像导致内存不足？

**A**: 这是正常的，因为 VAE 解码需要大量内存。解决方案：

1. 减少 `sampling_steps` 数量
2. 减少 `frame_num`
3. 使用更小的分辨率
4. 仅使用 `save_latents=True`，之后需要时再手动解码

### Q2: 如何只保存特定步骤的结果？

**A**: 目前实现会保存所有步骤。如果只需要特定步骤，可以：

1. 保存所有结果，然后删除不需要的
2. 加载后使用 Python 脚本过滤

### Q3: 可以恢复到生成过程中的某一步吗？

**A**: 可以。加载中间步骤的潜在代码，然后：

```python
# 从 step_010 继续生成
latents = torch.load("step_010_t0989/latents.pt")

# 创建新的采样器并继续采样
scheduler = FlowUniPCMultistepScheduler(...)
# ... 从第 11 步继续
```

### Q4: 为什么目录名中有两个时间步表示？

**A**: 
- `step_XXX`: 采样器的步数索引（0, 1, 2, ...）
- `t0YYY`: 扩散模型的实际时间步（1000, 999, 998, ...）

这样做可以：
- 清楚地标识生成顺序
- 了解实际的去噪时间步
- 便于重现或分析特定时间步

## 实现细节

中间结果保存由 `IntermediateResultSaver` 类处理（`utils/diffusion_utils.py`）：

```python
from wan.utils.diffusion_utils import IntermediateResultSaver

saver = IntermediateResultSaver(
    save_dir="./outputs",
    save_latents=True,
    save_decoded=False,
    vae=model.vae,  # 仅在 save_decoded=True 时需要
)

# 在每个扩散步调用
saver.save_step_results(
    latents=current_latents,
    step=timestep,
    step_idx=step_index,
    frame_num=81,
    vae_stride=(2, 8, 8),
)

# 最后显示总结
saver.get_summary()
```

## 最佳实践

### 1️⃣ 开发和调试
```python
# 快速验证，不保存中间结果
video = model.generate(
    ...,
    sampling_steps=20,  # 快速
    save_intermediate_dir=None,
)
```

### 2️⃣ 分析扩散过程
```python
# 保存潜在代码用于分析
video = model.generate(
    ...,
    sampling_steps=50,
    save_intermediate_dir="./analysis",
    save_latents=True,
    save_decoded=False,  # 之后需要时手动解码
)
```

### 3️⃣ 可视化生成过程
```python
# 保存所有内容用于可视化（需要充足的磁盘空间）
video = model.generate(
    ...,
    sampling_steps=30,   # 适中的步数
    save_intermediate_dir="./visualization",
    save_latents=True,
    save_decoded=True,   # 保存所有图像
)
```

### 4️⃣ 生产环境
```python
# 不保存中间结果以节省时间和空间
video = model.generate(
    ...,
    save_intermediate_dir=None,  # 禁用
)
```

## 故障排除

### 保存速度很慢？
- 检查磁盘 I/O 速度（使用 SSD）
- 禁用 `save_decoded`
- 减少采样步数

### 磁盘空间耗尽？
- 仅使用 `save_latents=True`
- 定期删除旧的中间结果
- 使用外部存储

### 内存不足 (OOM)？
- 禁用 `save_decoded`
- 减少 `frame_num`
- 减少批处理大小

## 相关资源

- 📄 使用示例：`examples/save_intermediate_results_example.py`
- 💾 实现代码：`wan/utils/diffusion_utils.py`
- 🎯 模型文件：`wan/text2video.py`, `wan/image2video.py` 等

---

**最后更新**: 2026年1月23日
