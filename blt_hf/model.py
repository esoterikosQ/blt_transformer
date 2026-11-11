"""Explicit model factory for the pinned B artifact and runtime semantics."""
from pathlib import Path
import json

from .manifest import verify_file_hashes


def configured_model(model_path, *, attention_mode, backend="eager"):
    from transformers import BltConfig
    if attention_mode not in ("osc", "lre") or backend not in ("eager", "sdpa"):
        raise ValueError("Explicit attention mode and supported backend required")
    config = BltConfig.from_pretrained(model_path, local_files_only=True)
    config.attention_mode = attention_mode
    config.use_cache = False
    config._attn_implementation = backend
    if attention_mode == "osc":
        spec = config.original_spec
        if spec["entropy_attn_bias_type"] != "local_block_causal" or spec["global_attn_bias_type"] != "block_causal":
            raise ValueError("Unreviewed original attention configuration")
    for name in ("patcher_config", "encoder_config", "decoder_config", "global_config"):
        child = getattr(config, name)
        child.attention_mode = attention_mode
        child.original_spec = config.original_spec
        child._attn_implementation = backend
    return config


def load_model(model_path, conversion_report, *, attention_mode, backend="eager", device="cuda"):
    import torch
    from .patched.modeling_blt import BltForCausalLM
    report = json.loads(Path(conversion_report).read_text())
    verify_file_hashes(model_path, report["output_files"])
    config = configured_model(model_path, attention_mode=attention_mode, backend=backend)
    model, info = BltForCausalLM.from_pretrained(model_path, config=config, local_files_only=True,
                                               dtype=torch.bfloat16, device_map=device,
                                               attn_implementation=backend, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Checkpoint does not strictly match model")
    return model
