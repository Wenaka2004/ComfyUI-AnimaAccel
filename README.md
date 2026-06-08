# ComfyUI-AnimaAccel

`torch.compile` acceleration node for the **Anima** text-to-image model (and any DiT-style diffusion model) in ComfyUI.

A thin, robust wrapper around ComfyUI's built-in `set_torch_compile_wrapper`, with portability fixes so it works across different environments out of the box.

> 中文说明见下方 / Chinese instructions below.

---

## What it does

Wraps the diffusion model's backbone with `torch.compile` (inductor backend). On an RTX 5090 this gives roughly **1.3–1.8×** faster denoising depending on mode and CUDA allocator, with **no quality change**.

The compile happens on the **first KSampler run** (takes a few minutes for `max-autotune`), then the compiled kernels are reused.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/<your-username>/ComfyUI-AnimaAccel
```
Restart ComfyUI.

## Usage

```
[Load Diffusion Model] → [Anima Compile] → [KSampler] → [VAE Decode]
```

Insert **Anima Compile** between the model loader and the sampler.

### Modes

| Mode | Speed | Notes |
|------|-------|-------|
| `max-autotune-no-cudagraphs` | ★★★★ | **Default.** Kernel autotuning, no CUDA graphs. Safe everywhere. |
| `max-autotune` | ★★★★★ | Adds CUDA graphs. Auto-downgrades on `cudaMallocAsync` allocator. |
| `reduce-overhead` | ★★★ | CUDA graphs only. Auto-downgrades on `cudaMallocAsync`. |
| `default` | ★★ | Basic inductor fusion. Fastest to compile. |

## Portability fixes (why this exists)

The plain `torch.compile` / built-in `TorchCompileModel` node crashes in several common setups. This node handles them automatically:

1. **`cudaMallocAsync` allocator** (used by some Windows builds, e.g. 秋叶/aki): CUDA graphs crash with `cudaMallocAsync does not yet support checkPoolLiveAllocations`. → Cudagraph modes are auto-downgraded **only when this allocator is detected**; native-allocator users keep full CUDA-graph speed.
2. **Windows without MSVC**: inductor CPU codegen fails with `Compiler: cl is not found`. → CPU codegen is disabled **only when no host compiler is present**.
3. **Dynamic-shape matmul error** in the DiT timestep embedder: → forces static shapes via `disable_dynamic`.

## Requirements

- ComfyUI recent enough to run Anima (provides `comfy_api.torch_helpers`)
- PyTorch ≥ 2.4 with a working Triton (bundled in standard ComfyUI installs)
- NVIDIA GPU

## Notes

- First run per resolution recompiles. Changing resolution triggers a new compile.
- If you want maximum speed and your build uses `cudaMallocAsync`, consider launching ComfyUI with the native allocator (`PYTORCH_CUDA_ALLOC_CONF=backend:native`) so the `max-autotune` mode can use CUDA graphs.

---

# 中文说明

ComfyUI 里给 **Anima** 文生图模型(以及任意 DiT 扩散模型)做 `torch.compile` 加速的节点。

是对 ComfyUI 自带 `set_torch_compile_wrapper` 的轻量封装,加了一堆可移植性修复,开箱即用。

## 效果

用 `torch.compile`(inductor 后端)编译扩散模型主干。RTX 5090 上去噪大约快 **1.3–1.8×**(取决于模式和 CUDA 分配器),**画质不变**。

编译发生在**第一次点 KSampler 时**(`max-autotune` 要等几分钟),之后复用编译好的 kernel。

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/<你的用户名>/ComfyUI-AnimaAccel
```
重启 ComfyUI。

## 用法

```
[Load Diffusion Model] → [Anima Compile] → [KSampler] → [VAE Decode]
```

把 **Anima Compile** 节点插在模型加载器和采样器之间。

### 模式

| 模式 | 速度 | 说明 |
|------|------|------|
| `max-autotune-no-cudagraphs` | ★★★★ | **默认。** kernel autotune,不用 CUDA graph,哪都能跑。 |
| `max-autotune` | ★★★★★ | 多开 CUDA graph。遇到 `cudaMallocAsync` 分配器自动降级。 |
| `reduce-overhead` | ★★★ | 只用 CUDA graph。遇到 `cudaMallocAsync` 自动降级。 |
| `default` | ★★ | 基础融合,编译最快。 |

## 为什么需要这个节点(可移植性修复)

裸 `torch.compile` 或自带的 `TorchCompileModel` 节点在几种常见环境里会崩,这个节点自动处理:

1. **`cudaMallocAsync` 分配器**(部分 Windows 整合包用,如秋叶):CUDA graph 会崩 `cudaMallocAsync does not yet support checkPoolLiveAllocations`。→ **只在检测到这个分配器时**自动降级 cudagraph 模式;native 分配器用户保留完整 CUDA graph 加速。
2. **Windows 没装 MSVC**:inductor CPU 代码生成报 `Compiler: cl is not found`。→ **只在找不到编译器时**关掉 CPU codegen。
3. **DiT timestep embedder 的动态形状 matmul 报错**:→ 用 `disable_dynamic` 强制静态形状。

## 依赖

- 能跑 Anima 的较新 ComfyUI(自带 `comfy_api.torch_helpers`)
- PyTorch ≥ 2.4,Triton 正常(标准 ComfyUI 都有)
- NVIDIA 显卡

## 提示

- 每个分辨率第一次跑会重新编译,换分辨率会触发新的编译。
- 如果你的整合包用 `cudaMallocAsync` 又想要最快速度,可以用 native 分配器启动 ComfyUI(`PYTORCH_CUDA_ALLOC_CONF=backend:native`),这样 `max-autotune` 就能用上 CUDA graph。
