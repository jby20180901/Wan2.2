# 🎬 Wan2.2 中间结果保存 - 快速参考

## 5秒快速上手

```python
video = model.generate(
    input_prompt="Your prompt",
    # ... 其他参数 ...
    save_intermediate_dir="./outputs",      # 启用保存
    save_latents=True,                       # 保存潜在代码
    save_decoded=False,                      # 不保存图像（太大）
)
```

## 参数速查表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_intermediate_dir` | str | None | 保存目录（None 为禁用） |
| `save_latents` | bool | True | 保存 .pt 文件（推荐） |
| `save_decoded` | bool | False | 保存 PNG 图像（谨慎使用） |

## 输出目录结构

```
save_intermediate_dir/
├── step_000_t0999/  ← 第0步，时间步999
│   ├── latents.pt   (100-200 MB)
│   └── frame_*.png  (可选，1-5 MB 每张)
├── step_001_t0998/  ← 第1步
│   └── latents.pt
└── final_video.pt
```

## 空间需求

| 配置 | 30步 | 50步 |
|------|------|------|
| latents only | ~3GB | ~5GB |
| + images | ~9GB | ~15GB |

## 💡 3个使用场景

### 1️⃣ 分析/调试 ✅
```python
generate(..., save_latents=True, save_decoded=False)
# 然后加载 latents 进行分析
```

### 2️⃣ 可视化生成过程 ✅  
```python
generate(..., 
    sampling_steps=20,      # 较少步数
    save_latents=True, 
    save_decoded=True       # 保存所有图像
)
```

### 3️⃣ 生产环境 ❌
```python
generate(..., save_intermediate_dir=None)  # 禁用
```

## 加载和使用

```python
import torch
from pathlib import Path

# 加载第 10 步的潜在代码
latents = torch.load("outputs/step_010_t0989/latents.pt")
# Shape: [C, T, H, W]

# 列出所有步骤
steps = sorted(Path("outputs").glob("step_*"))
for step in steps:
    print(step.name)  # step_000_t0999, step_001_t0998, ...
```

## 支持的模型

| 模型 | 文件 |
|------|------|
| 文本→视频 | `WanT2V` |
| 图像→视频 | `WanI2V` |
| 文本+图像→视频 | `WanTI2V` |
| 语音→视频 | `WanS2V` |
| 角色动画 | `WanAnimate` |

## 常见问题速答

**Q: 保存图像时内存不足？**  
A: 设置 `save_decoded=False`，只保存潜在代码

**Q: 生成太慢？**  
A: 禁用 `save_decoded`，或减少 `sampling_steps`

**Q: 如何从中间结果继续生成？**  
A: 加载潜在代码，从那一步继续采样

**Q: 命名中的数字是什么意思？**  
A: `step_XXX_tYYYY` - XXX是步数，YYYY是时间步（1000→0）

## 性能开销

| 操作 | 时间 | 空间 |
|------|------|------|
| 保存 latents | +0-1% | 最小 |
| 保存 images | +30-40% | 大量 |

## 相关文件

- 📖 详细指南: `INTERMEDIATE_RESULTS_GUIDE.md`
- 💻 代码示例: `examples/save_intermediate_results_example.py`
- 📝 实现: `wan/utils/diffusion_utils.py`

---

**提示**: 大多数情况下，`save_latents=True, save_decoded=False` 是最佳平衡
