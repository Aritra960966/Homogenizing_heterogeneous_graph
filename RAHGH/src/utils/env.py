import torch


def setup_training_env() -> None:
    """
    One-time GPU and threading configuration.
    Call once at process start, before any model or data initialisation.

    cudnn.benchmark
        Profiles cuDNN conv/matmul kernels on the first batch and caches
        the fastest implementation. Amortised cost: ~2s on first forward.
        Benefit: 5-15% speedup on GCN/GAT linear layers for fixed input sizes.
        Set False if input sizes vary across batches (they don't here).

    float32_matmul_precision = 'high'
        On Ampere+ GPUs (A100, RTX 30/40xx) this routes FP32 matmuls through
        Tensor Cores (TF32 internally, FP32 I/O). Typical speedup: 3-8x on
        large matmuls with <0.01% numerical difference.
        No effect on pre-Ampere GPUs or CPU.

    set_num_threads(1)
        Prevents PyTorch's intra-op thread pool from spawning multiple CPU
        threads that compete with the CUDA stream. On a GPU-bound workload
        the extra threads add context-switching overhead without benefit.
        Raise to os.cpu_count() // 2 only if you profile a CPU bottleneck.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    torch.set_num_threads(1)


def print_env_summary() -> None:
    """Print device and precision settings for run-log reproducibility."""
    print(f"[env] CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"[env] GPU            : {props.name}  "
              f"{props.total_memory // 1024**3} GB  "
              f"compute {props.major}.{props.minor}")
        print(f"[env] cudnn.benchmark: {torch.backends.cudnn.benchmark}")
        print(f"[env] matmul_precision: {torch.get_float32_matmul_precision()}")
    print(f"[env] num_threads    : {torch.get_num_threads()}")
