"""Load a converted BLT checkpoint with the settings verified in this project."""

ATTENTION_MODES = ("hf", "osc")


def load_model(path, *, attention_mode="hf", device="cuda", dtype=None,
               attn_implementation="eager", strict=True):
    """Return an eval-mode ``BltForCausalLM``.

    ``hf`` uses the stock Transformers class (full-causal local masks).
    ``osc`` uses ``blt_hf.patched`` to apply the original 512-window/EOS masks; it needs a
    checkpoint made by ``blt_transformer.convert`` and was verified only on transformers 5.16.1.
    """
    import torch
    if attention_mode not in ATTENTION_MODES:
        raise ValueError(f"attention_mode must be one of {ATTENTION_MODES}")
    if attention_mode == "osc":
        from blt_hf.model import configured_model
        from blt_hf.patched.modeling_blt import BltForCausalLM
        config = configured_model(path, attention_mode="osc", backend=attn_implementation)
    else:
        from transformers import BltConfig, BltForCausalLM
        config = BltConfig.from_pretrained(path)
        config.use_cache = False
    model, info = BltForCausalLM.from_pretrained(
        path, config=config, dtype=dtype or torch.bfloat16,
        attn_implementation=attn_implementation, output_loading_info=True)
    problems = {k: v for k, v in info.items() if v and k in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    if strict and problems:
        raise ValueError(f"Checkpoint does not strictly match BltForCausalLM: {problems}")
    return model.to(device).eval()
