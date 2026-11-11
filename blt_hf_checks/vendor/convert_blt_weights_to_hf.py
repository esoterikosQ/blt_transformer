import argparse
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# Project-contained caches. Never fall back to a user's global HF token/cache.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str(PROJECT_ROOT / "artifacts/hf_home")
os.environ["HF_HUB_CACHE"] = str(PROJECT_ROOT / "artifacts/hub")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from blt_hf.manifest import sha256_file, write_json
from blt_hf_checks.auth import read_project_token
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import bytes_to_unicode
from transformers.utils import logging as transformers_logging


logger = transformers_logging.get_logger(__name__)
transformers_logging.set_verbosity_info()


def merge_configurations(config_path: str, entropy_params_path: str) -> dict[str, Any]:
    logger.info("Merging configurations")

    with open(config_path, "r") as f:
        main_config = json.load(f)

    with open(entropy_params_path, "r") as f:
        entropy_data = json.load(f)

    entropy_model_params = entropy_data.get("entropy_model", {})
    patcher_args = entropy_data.get("data", {}).get("patcher_args", {})

    unified_config = main_config.copy()["args"]

    for key in ["vocab_size", "dim", "n_layers", "n_heads", "max_seqlen"]:
        if key in unified_config and not isinstance(unified_config[key], int):
            unified_config[key] = int(unified_config[key])

    patch_size = patcher_args.get("patch_size", 8)
    if isinstance(patch_size, float):
        patch_size = int(patch_size)

    # Create patcher config
    patcher_hidden_size = int(entropy_model_params.get("dim", 512))
    patcher_multiple_of = int(entropy_model_params.get("multiple_of", 256))
    patcher_intermediate_size = patcher_multiple_of * (
        (int(8 * patcher_hidden_size / 3) + patcher_multiple_of - 1) // patcher_multiple_of
    )

    patcher_config = {
        "vocab_size": int(entropy_model_params.get("vocab_size", 256)),
        "hidden_size": patcher_hidden_size,
        "num_hidden_layers": int(entropy_model_params.get("n_layers", 8)),
        "num_attention_heads": int(entropy_model_params.get("n_heads", 8)),
        "num_key_value_heads": int(entropy_model_params.get("n_kv_heads"))
        if entropy_model_params.get("n_kv_heads") is not None
        else None,
        "max_position_embeddings": int(entropy_model_params.get("max_seqlen", 1024)),
        "rms_norm_eps": entropy_model_params.get("norm_eps", 1e-5),
        "dropout": entropy_model_params.get("dropout", 0.0),
        "rope_theta": entropy_model_params.get("rope_theta", 10000.0),
        "rope_parameters": {"rope_type": "default", "rope_theta": entropy_model_params.get("rope_theta", 10000.0)},
        "attn_impl": entropy_model_params.get("attn_impl", "sdpa"),
        "attn_bias_type": entropy_model_params.get("attn_bias_type", "causal"),
        "intermediate_size": patcher_intermediate_size,
    }

    # Create encoder config
    encoder_hidden_size = unified_config.get("dim_local_encoder", 1024)
    encoder_multiple_of = unified_config.get("multiple_of", 256)
    encoder_intermediate_size = encoder_multiple_of * (
        (int(8 * encoder_hidden_size / 3) + encoder_multiple_of - 1) // encoder_multiple_of
    )

    encoder_config = {
        "vocab_size": unified_config.get("vocab_size", 256),
        "cross_attn_all_layers": unified_config.get("cross_attn_all_layers_encoder", False),
        "cross_attn_k": unified_config.get("cross_attn_k", 2),
        "hidden_size_global": unified_config.get("dim_global", 2048),
        "pm_size": unified_config.get("pm_size", 0),
        "hidden_size": encoder_hidden_size,
        "num_attention_heads": unified_config.get("n_heads_local_encoder", 16),
        "num_key_value_heads": unified_config.get("n_kv_heads"),
        "num_hidden_layers": unified_config.get("n_layers_local_encoder", 1),
        "rms_norm_eps": unified_config.get("norm_eps", 1e-5),
        "dropout": unified_config.get("dropout", 0.0),
        "max_position_embeddings": unified_config.get("max_encoder_seq_length")
        or unified_config.get("max_seqlen", 1024),
        "rope_theta": unified_config.get("rope_theta", 10000.0),
        "rope_parameters": {"rope_type": "default", "rope_theta": unified_config.get("rope_theta", 10000.0)},
        "hidden_act": unified_config.get("hidden_act", "silu"),
        "_attn_implementation": unified_config.get("_attn_implementation", "sdpa"),
        "intermediate_size": encoder_intermediate_size,
    }

    # Create decoder config
    decoder_hidden_size = unified_config.get("dim_local_decoder", 1024)
    decoder_multiple_of = unified_config.get("multiple_of", 256)
    decoder_intermediate_size = decoder_multiple_of * (
        (int(8 * decoder_hidden_size / 3) + decoder_multiple_of - 1) // decoder_multiple_of
    )

    decoder_config = {
        "vocab_size": unified_config.get("vocab_size", 256),
        "cross_attn_all_layers": unified_config.get("cross_attn_all_layers_decoder", False),
        "cross_attn_k": unified_config.get("cross_attn_k", 2),
        "hidden_size_global": unified_config.get("dim_global", 2048),
        "hidden_size": decoder_hidden_size,
        "num_attention_heads": unified_config.get("n_heads_local_decoder", 16),
        "num_key_value_heads": unified_config.get("n_kv_heads"),
        "num_hidden_layers": unified_config.get("n_layers_local_decoder", 9),
        "rms_norm_eps": unified_config.get("norm_eps", 1e-5),
        "dropout": unified_config.get("dropout", 0.0),
        "max_position_embeddings": unified_config.get("max_encoder_seq_length")
        or unified_config.get("max_seqlen", 1024),
        "rope_theta": unified_config.get("rope_theta", 10000.0),
        "rope_parameters": {"rope_type": "default", "rope_theta": unified_config.get("rope_theta", 10000.0)},
        "hidden_act": unified_config.get("hidden_act", "silu"),
        "_attn_implementation": unified_config.get("_attn_implementation", "sdpa"),
        "intermediate_size": decoder_intermediate_size,
    }

    # Create global transformer config
    global_hidden_size = unified_config.get("dim_global", 2048)
    global_multiple_of = unified_config.get("multiple_of", 256)
    global_intermediate_size = global_multiple_of * (
        (int(8 * global_hidden_size / 3) + global_multiple_of - 1) // global_multiple_of
    )

    global_config = {
        "hidden_size": global_hidden_size,
        "num_attention_heads": unified_config.get("n_heads_global", 16),
        "num_key_value_heads": unified_config.get("n_kv_heads_global"),
        "num_hidden_layers": unified_config.get("n_layers_global", 25),
        "rms_norm_eps": unified_config.get("norm_eps", 1e-5),
        "dropout": unified_config.get("dropout", 0.0),
        "max_position_embeddings": unified_config.get("max_seqlen", 1024),
        "rope_theta": unified_config.get("rope_theta", 10000.0),
        "rope_parameters": {"rope_type": "default", "rope_theta": unified_config.get("rope_theta", 10000.0)},
        "hidden_act": unified_config.get("hidden_act", "silu"),
        "_attn_implementation": unified_config.get("_attn_implementation", "sdpa"),
        "intermediate_size": global_intermediate_size,
    }

    # Create main config with sub-configs
    main_config_dict = {
        "model_type": "blt",
        "vocab_size": unified_config.get("vocab_size", 256),
        "max_position_embeddings": unified_config.get("max_seqlen", 1024),
        "patch_in_forward": True,
        "realtime_patching": True,
        "patching_mode": "entropy",
        "patch_size": patch_size,
        "patching_threshold": patcher_args.get("threshold", 0.5),
        "patching_threshold_add": patcher_args.get("threshold_add", 0.0),
        "max_patch_length": patcher_args.get("max_patch_length"),
        "patching_batch_size": patcher_args.get("patching_batch_size", 1),
        "patching_device": patcher_args.get("patching_device", "cuda"),
        "monotonicity": patcher_args.get("monotonicity", False),
        "cross_attn_k": unified_config.get("cross_attn_k", 2),
        "encoder_hash_byte_group_size": unified_config.get("encoder_hash_byte_group_size"),
        "encoder_hash_byte_group_vocab": unified_config.get("encoder_hash_byte_group_vocab", 30000),
        "encoder_hash_byte_group_nb_functions": unified_config.get("encoder_hash_byte_group_nb_functions", 3),
        "pm_size": unified_config.get("pm_size", 0),
        "patcher_config": patcher_config,
        "encoder_config": encoder_config,
        "decoder_config": decoder_config,
        "global_config": global_config,
    }

    main_config_dict["tie_word_embeddings"] = False
    main_config_dict.update(bos_token_id=1, eos_token_id=2, pad_token_id=3, use_cache=False)
    main_config_dict["original_spec"] = {
        "local_attention_window_len": unified_config["local_attention_window_len"],
        "local_attn_bias_type": "local_block_causal",
        "global_attn_bias_type": unified_config["attn_bias_type"],
        "entropy_attn_bias_type": entropy_model_params["attn_bias_type"],
        "entropy_sliding_window": entropy_model_params.get("sliding_window"),
        "eos_id": unified_config["eos_id"],
        "patch_size": unified_config["patch_size"],
        "patching_threshold": unified_config["patching_threshold"],
        "patching_threshold_add": unified_config.get("patching_threshold_add"),
        "monotonicity": unified_config.get("monotonicity", False),
        "max_patch_length": unified_config.get("max_patch_length"),
    }
    # HF's integer patch_size is only a sentinel for threshold-based patching.
    # Preserve the source's fractional target average separately; never present it as 4.
    if unified_config["patching_mode"] != "entropy" or unified_config.get("monotonicity", False):
        raise ValueError("Only the inspected entropy threshold policy is supported")
    if unified_config.get("patching_threshold_add") not in (None, 0, 0.0):
        raise ValueError("Additive entropy thresholds require a separate implementation")
    main_config_dict["patching_threshold"] = unified_config["patching_threshold"]

    logger.info(f"Merged configuration with {len(main_config_dict)} parameters")
    return main_config_dict


def apply_weight_mapping(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    component_mappings = {
        ".attention.": ".self_attn.",
        ".feed_forward.": ".mlp.",
        ".attention_norm.": ".input_layernorm.",
        ".ffn_norm.": ".post_attention_layernorm.",
        ".tok_embeddings.": ".embed_tokens.",
        ".cross_attn_norm_q.": ".q_norm.",
        ".cross_attn_norm_kv.": ".k_norm.",
        ".w1.": ".gate_proj.",
        ".w2.": ".down_proj.",
        ".w3.": ".up_proj.",
        ".wq.": ".q_proj.",
        ".wk.": ".k_proj.",
        ".wv.": ".v_proj.",
        ".wo.": ".o_proj.",
        ".output.": ".lm_head.",
    }

    new_state_dict = {}

    for old_key, tensor in state_dict.items():
        new_key = old_key

        for old_pattern, new_pattern in component_mappings.items():
            if old_pattern in new_key:
                new_key = new_key.replace(old_pattern, new_pattern)

        if new_key in new_state_dict:
            raise ValueError(f"Weight mapping collision: {old_key} -> {new_key}")
        new_state_dict[new_key] = tensor

    return new_state_dict


def convert_hash_embeddings_to_fused(
    unified_weights: dict[str, torch.Tensor], config: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Convert ModuleList hash embeddings to nn.embedding format"""
    original_keys_format = [
        key
        for key in unified_weights.keys()
        if "encoder_hash_tok_embedding." in key and ".weight" in key and key.split(".")[-2].isdigit()
    ]

    num_embeddings = config.get("encoder_hash_byte_group_nb_functions", 1) * len(
        config.get("encoder_hash_byte_group_size", [3, 4, 5, 6, 7, 8])
    )
    vocab_size = config.get("encoder_hash_byte_group_vocab", 500002)
    hidden_size = config.get("encoder_config", {}).get("hidden_size", 1024)

    sorted_keys = sorted(original_keys_format, key=lambda k: int(k.split(".")[-2]))
    indices = [int(key.split(".")[-2]) for key in sorted_keys]
    if indices != list(range(num_embeddings)):
        raise ValueError(f"Missing or unexpected hash embedding slices: {indices}")
    original = unified_weights[sorted_keys[0]]
    for key in sorted_keys:
        tensor = unified_weights[key]
        if tensor.shape != (vocab_size, hidden_size) or tensor.dtype != original.dtype:
            raise ValueError(f"Inconsistent hash embedding shape/dtype: {key}")
    fused_weight = torch.empty(vocab_size * num_embeddings, hidden_size, dtype=original.dtype)

    for i, old_key in enumerate(sorted_keys):
        start_idx = i * vocab_size
        end_idx = (i + 1) * vocab_size
        fused_weight[start_idx:end_idx] = unified_weights[old_key]
        logger.info(f"Copied {old_key} to indices {start_idx}:{end_idx}")
        del unified_weights[old_key]

    fused_key = "model.encoder_hash_tok_embedding.weight"
    unified_weights[fused_key] = fused_weight

    return unified_weights


def merge_weights(weights_path: str, entropy_weights_path: str) -> dict[str, torch.Tensor]:
    main_weights = load_file(weights_path)

    entropy_weights = (load_file(entropy_weights_path) if str(entropy_weights_path).endswith(".safetensors")
                       else torch.load(entropy_weights_path, map_location="cpu", weights_only=True))

    if "model" in entropy_weights:
        entropy_weights = entropy_weights["model"]
    elif "state_dict" in entropy_weights:
        entropy_weights = entropy_weights["state_dict"]

    unified_weights = main_weights.copy()

    for key, tensor in entropy_weights.items():
        patcher_key = f"patcher.{key}"
        unified_weights[patcher_key] = tensor

    unified_weights = apply_weight_mapping(unified_weights)

    decoder_lm_head_key = "local_decoder.lm_head.weight"
    top_lm_head_key = "lm_head.weight"
    unified_weights[top_lm_head_key] = unified_weights[decoder_lm_head_key]
    del unified_weights[decoder_lm_head_key]

    prefixed_weights = {}
    for key, tensor in unified_weights.items():
        if key == top_lm_head_key:
            prefixed_weights[key] = tensor
        elif not key.startswith("model."):
            prefixed_weights[f"model.{key}"] = tensor
        else:
            prefixed_weights[key] = tensor

    unified_weights = prefixed_weights

    return unified_weights


def create_tokenizer_config(output_dir: str, config: dict[str, Any]):
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "vocab_size": config.get("vocab_size", 256),
        "model_max_length": config["max_position_embeddings"],
        "model_input_names": ["input_ids", "attention_mask"],
        "add_bos_token": False,
        "add_eos_token": False,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }

    tokenizer_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)


def create_tokenizer_json(output_dir: str, config: dict[str, Any]):
    byte_encoder = bytes_to_unicode()

    vocab: dict[str, int] = {}
    vocab["<boe>"] = 0
    vocab["<s>"] = 1
    vocab["</s>"] = 2
    vocab["<pad>"] = 3

    offset = 4
    for byte_val, unicode_char in byte_encoder.items():
        vocab[unicode_char] = byte_val + offset

    backend = Tokenizer(
        BPE(vocab=vocab, merges=[], continuing_subword_prefix="", end_of_word_suffix="", fuse_unk=False)
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()

    # Explicit special-token ownership belongs to the data adapter.
    backend.post_processor = processors.TemplateProcessing(single="$A:0", pair="$A:0 $B:1")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=config.get("bos_token", "<s>"),
        eos_token=config.get("eos_token", "</s>"),
        pad_token=config.get("pad_token", "<pad>"),
        unk_token=config.get("unk_token", "<unk>"),
    )

    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    tokenizer.model_max_length = config["max_position_embeddings"]
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved tokenizer.json to {os.path.join(output_dir, 'tokenizer.json')}")



def require_revision(value):
    if not re.fullmatch(r"[a-fA-F0-9]{40}", value or ""):
        raise ValueError("revision must be a full 40-character commit SHA")
    return value.lower()


def project_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path must be within the project: {path}")
    return path


def source_file(repo, revision, filename, local_root, token, offline):
    require_revision(revision)
    if local_root:
        path = project_path(Path(local_root) / filename)
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    return hf_hub_download(repo_id=repo, revision=revision, filename=filename,
                           cache_dir=str(PROJECT_ROOT / "artifacts/hub"),
                           token=token, local_files_only=offline)


def main():
    parser = argparse.ArgumentParser(description="Pinned, project-contained BLT conversion; CPU only")
    parser.add_argument("--model_id", default="facebook/blt-1b")
    parser.add_argument("--model-revision", required=True, type=require_revision)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source-dir", help="Optional project-local main snapshot with original file layout")
    parser.add_argument("--entropy-model-id")
    parser.add_argument("--entropy-revision", type=require_revision)
    parser.add_argument("--entropy-source-dir")
    parser.add_argument("--token-file", help="Project-local HF token file; contents are never logged")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--report", required=True, help="New conversion evidence JSON under project")
    args = parser.parse_args()
    if bool(args.entropy_model_id) != bool(args.entropy_revision):
        parser.error("--entropy-model-id and --entropy-revision must be supplied together")
    if args.entropy_source_dir and not args.entropy_model_id:
        parser.error("--entropy-source-dir requires an entropy model ID and revision")
    output, report_path = project_path(args.output_dir), project_path(args.report)
    if output.exists() or report_path.exists():
        parser.error("Output/report already exists; preserve evidence and choose new paths")
    token = os.environ.get("HF_TOKEN") or False
    if args.token_file:
        token = read_project_token(args.token_file, PROJECT_ROOT)
    inputs = {}
    for name in ("config.json", "model.safetensors", "entropy_model/params.json", "entropy_model/consolidated.pth"):
        inputs[name] = source_file(args.model_id, args.model_revision, name, args.source_dir, token, args.local_files_only)
    inputs["LICENSE"] = source_file(args.model_id, args.model_revision, "LICENSE", args.source_dir, token, args.local_files_only)
    entropy_weights = inputs["entropy_model/consolidated.pth"]
    entropy_params = inputs["entropy_model/params.json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".blt-conversion-", dir=output.parent) as staging:
        if args.entropy_model_id:
            for name in ("config.json", "model.safetensors"):
                inputs["legacy_entropy/" + name] = source_file(args.entropy_model_id, args.entropy_revision, name,
                                                               args.entropy_source_dir, token, args.local_files_only)
            legacy_config = json.loads(Path(inputs["legacy_entropy/config.json"]).read_text())
            if not isinstance(legacy_config.get("args"), dict):
                raise ValueError("Unknown legacy entropy config schema: expected config.args; inspect source before adapting")
            params = json.loads(Path(entropy_params).read_text())
            params["entropy_model"] = legacy_config["args"]
            entropy_params = str(Path(staging) / "selected_entropy_params.json")
            Path(entropy_params).write_text(json.dumps(params))
            entropy_weights = inputs["legacy_entropy/model.safetensors"]
        config = merge_configurations(inputs["config.json"], entropy_params)
        if config["max_position_embeddings"] < 2048:
            raise ValueError("Original model context is below 2048; do not silently increase it")
        weights = merge_weights(inputs["model.safetensors"], entropy_weights)
        weights = convert_hash_embeddings_to_fused(weights, config)
        result_dir = Path(staging) / "model"
        result_dir.mkdir()
        (result_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (result_dir / "LICENSE").write_bytes(Path(inputs["LICENSE"]).read_bytes())
        (result_dir / "README.md").write_text(
            "# P1 direct BLT conversion B\n\n"
            "Original weights: facebook/blt-1b; source license is reproduced in LICENSE.\n"
            "Internal research artifact. Identity validation is pending.\n"
            "The original_spec config records source semantics; stock HF execution does not implement all of them.\n"
        )
        save_file(weights, str(result_dir / "model.safetensors"))
        create_tokenizer_json(str(result_dir), config)
        create_tokenizer_config(str(result_dir), config)
        report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                  "status": "converted_unvalidated", "validation_status": "pending",
                  "model_id": args.model_id, "model_revision": args.model_revision,
                  "entropy_model_id": args.entropy_model_id or args.model_id,
                  "entropy_revision": args.entropy_revision or args.model_revision,
                  "converter_upstream_commit": "2cba19507be799b7bef247ca6c1c4708bf881b5b",
                  "converter_sha256": sha256_file(__file__),
                  "inputs": {name: {"path": path, "sha256": sha256_file(path)} for name, path in inputs.items()},
                  "output_files": {p.name: sha256_file(p) for p in result_dir.iterdir() if p.is_file()},
                  "tensor_count": len(weights), "tensor_dtypes": sorted({str(t.dtype) for t in weights.values()}),
                  "full_weight_parity_passed": False, "forward_parity_passed": False}
        if args.entropy_model_id:
            report["entropy_serialized_files_equal"] = (report["inputs"]["entropy_model/consolidated.pth"]["sha256"] ==
                                                        report["inputs"]["legacy_entropy/model.safetensors"]["sha256"])
            report["entropy_note"] = "Different serialization hashes do not prove different tensors; tensor/config parity still pending"
        write_json(report_path, report)
        result_dir.rename(output)
    print(json.dumps({"status": report["status"], "output": str(output), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
