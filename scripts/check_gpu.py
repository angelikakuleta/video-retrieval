"""
Environment check: does the pipeline actually compute on the GPU?

Run once after installing the environment:
    python check_gpu.py

The script checks TWO independent execution engines:
  1. PyTorch  -- OpenCLIP, LLaVA, BLIP, X-CLIP, YOLO, SlowFast, TransNetV2
  2. ONNX Runtime -- RetinaFace R50 + ArcFace glintr100

These two engines have separate CUDA builds and can fail independently of each other.
"""

EXPECTED_ARCH = "sm_120"        # RTX 5080, Blackwell architecture
MIN_ONNXRUNTIME = (1, 27)       # below this no sm_120 kernels -> silent fallback to CPU

results = []


def record(name, ok, message):
    results.append((name, ok, message))
    mark = "OK  " if ok else "FAIL"
    print(f"[{mark}] {name}: {message}")


# ---------------------------------------------------------------- 1. PyTorch
print("\n=== PyTorch ===")
try:
    import torch

    print(f"       torch version: {torch.__version__}, CUDA: {torch.version.cuda}")

    if not torch.cuda.is_available():
        record("PyTorch sees the GPU", False,
               "torch.cuda.is_available() == False. Most common cause: "
               "a CPU-only build from PyPI is installed. Reinstall with "
               "--index-url https://download.pytorch.org/whl/cu130")
    else:
        record("PyTorch sees the GPU", True, torch.cuda.get_device_name(0))

        arch = torch.cuda.get_arch_list()
        if EXPECTED_ARCH in arch:
            record(f"Kernels for {EXPECTED_ARCH}", True, f"present ({', '.join(arch)})")
        else:
            record(f"Kernels for {EXPECTED_ARCH}", False,
                   f"MISSING. Available: {', '.join(arch)}. Wrong CUDA variant "
                   f"(cu126 does not include sm_120 -- cu130 is needed)")

        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        record("GPU memory", True, f"{vram:.1f} GB")

        # a real computation + timing: does the GPU compute and does it use tensor cores
        import time

        def benchmark(dtype, n=4096, repeats=20):
            a = torch.randn(n, n, device="cuda", dtype=dtype)
            b = torch.randn(n, n, device="cuda", dtype=dtype)
            for _ in range(3):                      # warm-up
                a @ b
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(repeats):
                a @ b
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / repeats
            tflops = 2 * n**3 / elapsed / 1e12
            del a, b
            torch.cuda.empty_cache()
            return elapsed, tflops

        time32, tflops32 = benchmark(torch.float32)
        time16, tflops16 = benchmark(torch.float16)
        print(f"       matmul 4096x4096: fp32 {time32*1000:6.1f} ms ({tflops32:5.1f} TFLOPS)")
        print(f"                         fp16 {time16*1000:6.1f} ms ({tflops16:5.1f} TFLOPS)")

        # an RTX 5080 class GPU should give well above 10 TFLOPS in fp32;
        # a result of single TFLOPS means the CPU is computing
        if tflops32 < 5:
            record("Performance", False,
                   f"only {tflops32:.1f} TFLOPS in fp32 -- that is CPU level, "
                   f"not GPU. Check the points above")
        else:
            record("Performance", True, f"{tflops32:.1f} TFLOPS (fp32), "
                                        f"{tflops16:.1f} TFLOPS (fp16)")
            if tflops16 > tflops32 * 1.5:
                print(f"       -> tensor cores work (fp16 is "
                      f"{tflops16/tflops32:.1f}x faster than fp32)")

except ImportError as e:
    record("PyTorch", False, f"package missing ({e})")
except Exception as e:
    record("PyTorch", False, f"{type(e).__name__}: {e}")


# ------------------------- 2. ONNX Runtime -- separate engine, separate risk
print("\n=== ONNX Runtime (face detection and recognition) ===")
try:
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        # ORT >= 1.21: loads the cublas/cudnn DLLs from the pip nvidia-* packages.
        # Without it a session created before importing torch silently falls
        # back to the CPU ("cublasLt64_13.dll is missing").
        ort.preload_dlls()

    version = ort.__version__
    version_tuple = tuple(int(x) for x in version.split(".")[:2])
    print(f"       onnxruntime version: {version}")

    if version_tuple < MIN_ONNXRUNTIME:
        record("onnxruntime version", False,
               f"{version} < {'.'.join(map(str, MIN_ONNXRUNTIME))}. Builds up to 1.26 "
               f"have no sm_120 kernels and SILENTLY compute on the CPU. "
               f"pip install -U \"onnxruntime-gpu[cuda,cudnn]\"")
    else:
        record("onnxruntime version", True, version)

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        record("CUDAExecutionProvider", True, "available")
    else:
        record("CUDAExecutionProvider", False,
               f"UNAVAILABLE. Visible: {available}. Most likely the plain "
               f"onnxruntime is installed instead of onnxruntime-gpu. Note: these two "
               f"packages cannot coexist -- uninstall onnxruntime, then install "
               f"onnxruntime-gpu")

except ImportError as e:
    record("ONNX Runtime", False, f"package missing ({e})")
except Exception as e:
    record("ONNX Runtime", False, f"{type(e).__name__}: {e}")


# --------------------------- 3. face models (RetinaFace + ArcFace glintr100)
print("\n=== face models (RetinaFace R50 + ArcFace glintr100) ===")
try:
    from pathlib import Path

    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        # ORT >= 1.21: loads the cublas/cudnn DLLs from the pip nvidia-* packages.
        # Without it a session created before importing torch silently falls
        # back to the CPU ("cublasLt64_13.dll is missing").
        ort.preload_dlls()

    ROOT = Path(__file__).resolve().parents[1]
    FACE_MODELS = {
        "retinaface": ROOT / "data" / "models" / "retinaface_r50.onnx",
        "arcface": Path.home() / ".insightface" / "models" / "antelopev2" / "glintr100.onnx",
    }

    on_cpu, missing = [], []
    for name, path in FACE_MODELS.items():
        if not path.exists():
            missing.append(f"{name} ({path})")
            continue
        session = ort.InferenceSession(
            str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        active = session.get_providers()[0]
        print(f"       {name:<12} -> {active}")
        if active != "CUDAExecutionProvider":
            on_cpu.append(name)

    if missing:
        record("Face models present", False,
               "missing files: " + "; ".join(missing)
               + " - each raises with the steps to produce it when first loaded")
    if on_cpu:
        record("Face models on the GPU", False,
               f"computing on the CPU: {', '.join(on_cpu)}")
    elif not missing:
        record("Face models on the GPU", True, "all on CUDA")

except ImportError as e:
    record("face models", False, f"package missing ({e})")
except Exception as e:
    record("face models", False, f"{type(e).__name__}: {e}")

# ------------------------------------------------------------------- Summary
print("\n" + "=" * 70)
failures = [(n, m) for n, ok, m in results if not ok]
if not failures:
    print("ALL OK -- the whole pipeline computes on the GPU.")
else:
    print(f"PROBLEMS ({len(failures)}):")
    for name, message in failures:
        print(f"  - {name}: {message}")
print("=" * 70)
print("""
A final note: even with a correct configuration you will see a busy CPU and
a partly idle GPU during processing. This is normal -- video decoding
(FFmpeg/OpenCV) is a CPU task and is usually the bottleneck of the pipeline.
It is not a symptom of a misconfiguration.
""")
