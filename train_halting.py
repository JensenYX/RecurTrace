import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from recurtrace import load_tokenizer
from recurtrace.loading import read_adapter_config


def main():
    parser = argparse.ArgumentParser(
        description="Distill a halting head from offline oracle targets."
    )
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", default="outputs/stage2")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-balance", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("steps, batch size and learning rate must be positive")
    if not 0 <= args.warmup_ratio <= 1 or args.weight_decay < 0:
        parser.error("invalid warmup ratio or weight decay")
    adapter_path, oracle_path = Path(args.adapter), Path(args.oracle)
    if adapter_path.resolve() == Path(args.output).resolve():
        parser.error("use a separate output directory for the trained halting adapter")
    config = read_adapter_config(adapter_path)
    with (oracle_path / "oracle_config.json").open(encoding="utf-8") as handle:
        oracle_config = json.load(handle)
    for key in ("base_config", "model"):
        if config[key] != oracle_config[key]:
            raise ValueError(f"oracle and adapter disagree on {key}")
    state = load_file(str(adapter_path / "adapter.safetensors"))
    oracle = load_file(str(oracle_path / "oracle.safetensors"))
    features, targets = oracle["features"].float(), oracle["targets"].float()
    if features.ndim != 3 or targets.shape != features.shape[:2] or not features.shape[0]:
        raise ValueError("invalid oracle feature or target shape")
    if features.shape[-1] != config["base_config"]["hidden_size"]:
        raise ValueError("oracle features do not match the adapter hidden size")
    if not bool(torch.isfinite(features).all()) or not bool(
        ((targets == 0) | (targets == 1)).all()
    ):
        raise ValueError("oracle features must be finite and targets must be binary")
    torch.manual_seed(args.seed)
    head = nn.Linear(features.shape[-1], 1).to(args.device)
    nn.init.normal_(head.weight, mean=0.0, std=0.02)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    warmup = round(args.steps * args.warmup_ratio)

    def schedule(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, args.steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    positives = targets.sum(dim=0)
    negatives = len(targets) - positives
    positive_weight = torch.where(
        (positives > 0) & (negatives > 0),
        negatives / positives.clamp_min(1),
        torch.ones_like(positives),
    ).to(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(features), generator=generator)
    cursor = 0
    for step in range(args.steps):
        if cursor >= len(order):
            order = torch.randperm(len(features), generator=generator)
            cursor = 0
        indices = order[cursor : cursor + args.batch_size]
        cursor += len(indices)
        logits = head(features[indices].to(args.device)).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets[indices].to(args.device),
            pos_weight=None if args.no_balance else positive_weight,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("halting loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(f"Step {step + 1}/{args.steps}: loss={loss.item():.6f}", flush=True)
    state["halting_head.weight"] = head.weight.detach().cpu().contiguous()
    state["halting_head.bias"] = head.bias.detach().cpu().contiguous()
    config["halting_trained"] = True
    config["halting"] = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "balanced_bce": not args.no_balance,
        "seed": args.seed,
        "probe_depths": oracle_config["probe_depths"],
        "margin": oracle_config["margin"],
        "oracle_max_depth": max(oracle_config["depths"]),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output / "adapter.safetensors"))
    with (output / "recurtrace_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    load_tokenizer(adapter=adapter_path).save_pretrained(output)
    print(f"Halting adapter saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
