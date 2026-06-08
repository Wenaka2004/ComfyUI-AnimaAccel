"""
ComfyUI-AnimaAccel — torch.compile acceleration for Anima (and any DiT) model.

Faithfully follows ComfyUI's built-in TorchCompileModel node, plus:
  - disable_dynamic=True to avoid dynamic-shape matmul errors
  - guard filter to skip transformer_options (avoids recompiles)
  - max-autotune-no-cudagraphs as default safe-fast mode
  - auto-downgrades CUDA-graph modes on cudaMallocAsync allocator
  - disables C++ codegen when no host compiler (e.g. Windows w/o MSVC)

Usage:  [Load Checkpoint] -> [Anima Compile] -> [KSampler] -> ...
"""
import os
import shutil
import sys
import torch


# --- Environment checks ---

def _has_host_compiler() -> bool:
    if sys.platform == "win32":
        return shutil.which("cl") is not None
    return any(shutil.which(c) for c in ("gcc", "g++", "clang", "clang++"))

if not _has_host_compiler():
    try:
        torch._inductor.config.cpp.enable = False
        print("[AnimaAccel] No host C++ compiler found — inductor CPU codegen disabled")
    except Exception:
        pass

from comfy_api.torch_helpers import set_torch_compile_wrapper


_CUDAGRAPH_MODES = {"max-autotune", "reduce-overhead"}


def _allocator_is_malloc_async() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.get_allocator_backend() == "cudaMallocAsync"
    except Exception:
        return False


def skip_torch_compile_dict(guard_entries):
    return [("transformer_options" not in entry.name) for entry in guard_entries]


# --- Node ---

class AnimaCompile:
    """Compile the diffusion model backbone with torch.compile."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["max-autotune-no-cudagraphs", "max-autotune",
                          "reduce-overhead", "default"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "AnimaAccel"

    def patch(self, model, mode):
        chosen = mode

        if mode in _CUDAGRAPH_MODES and _allocator_is_malloc_async():
            chosen = "max-autotune-no-cudagraphs" if mode == "max-autotune" else "default"
            print(f"[AnimaAccel] cudaMallocAsync allocator — downgrading "
                  f"'{mode}' to '{chosen}' (CUDA graphs unsupported here)")

        m = model.clone(disable_dynamic=True)

        opts = {"guard_filter_fn": skip_torch_compile_dict}
        if chosen in ("max-autotune", "max-autotune-no-cudagraphs"):
            opts["max_autotune"] = True
        use_cudagraphs = chosen in _CUDAGRAPH_MODES
        opts["triton.cudagraphs"] = use_cudagraphs
        if not use_cudagraphs:
            opts["triton.cudagraph_trees"] = False

        set_torch_compile_wrapper(model=m, backend="inductor", dynamic=False, options=opts)
        print(f"[AnimaAccel] armed: mode={chosen} max_autotune={opts.get('max_autotune',False)} "
              f"cudagraphs={use_cudagraphs}. Compiles on first KSampler run.")
        return (m,)


NODE_CLASS_MAPPINGS = {"AnimaCompile": AnimaCompile}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaCompile": "Anima Compile (torch.compile)"}
print("[AnimaAccel] Loaded")
