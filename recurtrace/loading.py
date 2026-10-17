import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen3Config

from .modeling import (
    DEFAULT_CONFIG,
    LOOP_PARAMETER_PREFIXES,
    RecurTraceForCausalLM,
    configure_model,
)


BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
ADAPTER_PREFIXES = (*LOOP_PARAMETER_PREFIXES, "halting_head.")


def read_adapter_config(path):
    with (Path(path) / "recurtrace_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("format_version") != 1:
        raise ValueError("unsupported RecurTrace adapter format")
    return config


def resolve_base_model(base_model=None, adapter=None):
    if base_model is not None:
        return str(base_model)
    if adapter is not None:
        base_model = read_adapter_config(adapter).get("base_model")
        if base_model:
            return base_model
        raise ValueError("pass --model with the base model used to train this adapter")
    return BASE_MODEL


def load_tokenizer(base_model=None, adapter=None):
    source = Path(adapter) if adapter is not None else None
    if source is None or not (source / "tokenizer_config.json").is_file():
        source = resolve_base_model(base_model, adapter)
    tokenizer = AutoTokenizer.from_pretrained(source)
    if not tokenizer.chat_template:
        raise ValueError("the tokenizer must provide a Qwen3 chat template")
    if tokenizer.eos_token_id is None:
        raise ValueError("the tokenizer must provide an end-of-sequence token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_adapter(model, path):
    config = read_adapter_config(path)
    state = load_file(str(Path(path) / "adapter.safetensors"))
    expected = {
        key: value for key, value in model.state_dict().items() if key.startswith(ADAPTER_PREFIXES)
    }
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    mismatched = sorted(
        key for key in expected.keys() & state.keys() if expected[key].shape != state[key].shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f"invalid adapter: missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    for key, value in config.get("base_config", {}).items():
        if getattr(model.config, key, None) != value:
            raise ValueError(f"adapter and base model disagree on {key}")
    model.load_state_dict(state, strict=False)
    model.halting_trained = bool(config.get("halting_trained", False))
    return model


def load_model(
    base_model=None, adapter=None, device="cpu", dtype=None, attention="sdpa", **overrides
):
    base_model = resolve_base_model(base_model, adapter)
    device = torch.device(device)
    if dtype is None:
        dtype = (
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
    config = Qwen3Config.from_pretrained(base_model)
    if adapter is not None:
        saved = read_adapter_config(adapter)
        overrides = {**saved["model"], **overrides}
    configure_model(config, **overrides)
    model = RecurTraceForCausalLM.from_pretrained(
        base_model, config=config, dtype=dtype, attn_implementation=attention
    )
    if adapter is not None:
        load_adapter(model, adapter)
    else:
        model.model.reset_recurtrace_parameters()
    model.base_model_reference = None if Path(base_model).exists() else base_model
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(LOOP_PARAMETER_PREFIXES))
    return model.to(device).eval()


def save_adapter(model, output_dir, tokenizer=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if key.startswith(ADAPTER_PREFIXES)
    }
    save_file(state, str(output_dir / "adapter.safetensors"))
    config = {
        "format_version": 1,
        "base_model": getattr(model, "base_model_reference", None),
        "base_config": {
            key: getattr(model.config, key)
            for key in (
                "model_type",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "tie_word_embeddings",
            )
        },
        "model": {key: getattr(model.config, key) for key in ("loop_start_layer", *DEFAULT_CONFIG)},
        "halting_trained": bool(model.halting_trained),
    }
    with (output_dir / "recurtrace_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    if tokenizer is not None:
        tokenizer.init_kwargs.pop("name_or_path", None)
        tokenizer.save_pretrained(output_dir)
