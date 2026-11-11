"""Inspect an existing environment. Never installs packages or changes drivers.

GPU mode performs a tiny CUDA matmul, with no optimizer or backward operation.
CPU mode reports metadata only and cannot certify a GPU environment.
"""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blt_hf.manifest import write_json

EXPECTED = {"torch": "2.11.0+cu128", "transformers": "5.16.1", "cuda_runtime": "12.8"}
SUPPORTED_BF16_CAPABILITIES = {
    (8, 0): "Ampere/A100",
    (9, 0): "Hopper/H100/H200",
    (12, 0): "Blackwell/RTX 5090",
}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_gpu_report(report: dict) -> list[str]:
    errors = list(report.get("errors", []))
    for key, expected in EXPECTED.items():
        if report.get(key) != expected:
            errors.append(f"{key}: expected {expected}, got {report.get(key)!r}")
    if not report.get("python", "").startswith("3.11."):
        errors.append("Python 3.11 is required by the GPU environment plan")
    for key in ("cuda_available", "blt_import", "cuda_smoke"):
        if report.get(key) is not True:
            errors.append(f"{key}: check did not pass")
    if report.get("xformers") is not None:
        errors.append("xformers must be absent from phdq_blt_hf")
    devices = report.get("devices", [])
    if not devices:
        errors.append("No CUDA devices")
    for device in devices:
        if tuple(device.get("capability", ())) not in SUPPORTED_BF16_CAPABILITIES:
            errors.append(f"Unverified GPU capability: {device}")
        if device.get("bf16_supported") is not True:
            errors.append(f"Native BF16 support check failed: {device}")
        if device.get("bf16_smoke") is not True:
            errors.append(f"BF16 CUDA matmul check failed: {device}")
    return errors


def inspect_environment(cpu_only=False):
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "hostname": platform.node(), "platform": platform.platform(),
              "python": platform.python_version(), "python_executable": sys.executable,
              "scope": "cpu_metadata_only" if cpu_only else "gpu_runtime",
              **{name: package_version(name) for name in ("torch", "transformers", "xformers", "safetensors", "huggingface_hub")},
              "devices": [], "errors": []}
    if cpu_only:
        report["status"] = "informational"
        return report
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=15, check=True)
        report["nvidia_smi"] = result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        report["errors"].append(f"nvidia-smi: {exc}")
    try:
        import torch
        from transformers import BltConfig, BltForCausalLM
        report.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                      blt_import=True, cuda_available=torch.cuda.is_available(), cuda_smoke=False)
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            with torch.no_grad(), torch.cuda.device(index):
                bf16_supported = bool(torch.cuda.is_bf16_supported())
                value = torch.ones((16, 16), device=f"cuda:{index}", dtype=torch.bfloat16)
                product = value @ value
                torch.cuda.synchronize(index)
                bf16_smoke = product.dtype == torch.bfloat16 and torch.all(product == 16).item()
                if not bf16_smoke:
                    raise RuntimeError(f"BF16 CUDA matmul failed on device {index}")
            report["devices"].append({"index": index, "name": props.name,
                                      "capability": list(torch.cuda.get_device_capability(index)),
                                      "total_memory": props.total_memory,
                                      "bf16_supported": bf16_supported, "bf16_smoke": bf16_smoke})
        report["cuda_smoke"] = bool(report["devices"])
        maps = Path("/proc/self/maps")
        if maps.exists():
            report["loaded_libcudart"] = sorted({line.split()[-1] for line in maps.read_text().splitlines() if "libcudart" in line})
    except Exception as exc:
        report["errors"].append(f"runtime: {type(exc).__name__}: {exc}")
    report["errors"] = validate_gpu_report(report)
    report["status"] = "pass" if not report["errors"] else "fail"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--output", type=Path, help="New immutable JSON report (existing file is an error)")
    args = parser.parse_args()
    report = inspect_environment(args.cpu_only)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
