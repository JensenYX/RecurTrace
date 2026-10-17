import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from recurtrace import load_model, load_tokenizer
from recurtrace.data import answer_nll, encode_batch, load_data
from recurtrace.loading import read_adapter_config


def main():
    parser = argparse.ArgumentParser(
        description="Collect loop states and deeper-loss targets for halting."
    )
    parser.add_argument("--model")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs/oracle")
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--probe-depths", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    args = parser.parse_args()
    probe_depths = sorted(set(args.probe_depths))
    if not probe_depths or probe_depths[0] < 1 or probe_depths[-1] >= args.max_depth:
        parser.error("probe depths must be positive and below --max-depth")
    if args.batch_size < 1 or args.max_length < 2 or not 0 <= args.margin < float("inf"):
        parser.error("invalid batch size, sequence length, or margin")
    tokenizer = load_tokenizer(args.model, args.adapter)
    model = load_model(args.model, args.adapter, device=args.device, attention=args.attention)
    dataset = load_data(args.data, seed=args.seed, limit=args.limit)
    depths = list(range(1, args.max_depth + 1))
    features, losses = [], []
    with torch.inference_mode():
        for offset in range(0, len(dataset), args.batch_size):
            rows = list(dataset.select(range(offset, min(offset + args.batch_size, len(dataset)))))
            batch = encode_batch(tokenizer, rows, args.max_length, args.device)
            model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                prompt_mask=batch["prompt_mask"],
                collect_depths=depths,
                use_cache=False,
            )
            batch_losses = []
            for depth in depths:
                hidden = model.model.collected_depth_hidden.pop(depth)
                batch_losses.append(answer_nll(model.lm_head(hidden), batch["labels"]).cpu())
            losses.append(torch.stack(batch_losses, dim=1))
            features.append(
                torch.stack(
                    [
                        model.model.collected_decision_features[depth].float().cpu()
                        for depth in probe_depths
                    ],
                    dim=1,
                )
            )
            model.model.collected_decision_features.clear()
            print(
                f"Oracle {min(offset + args.batch_size, len(dataset))}/{len(dataset)}", flush=True
            )
    loss_tensor = torch.cat(losses).float()
    targets = torch.stack(
        [
            (
                loss_tensor[:, depth:].min(dim=1).values < loss_tensor[:, depth - 1] - args.margin
            ).float()
            for depth in probe_depths
        ],
        dim=1,
    )
    feature_tensor = torch.cat(features)
    if not bool(torch.isfinite(loss_tensor).all() and torch.isfinite(feature_tensor).all()):
        raise RuntimeError("oracle collection produced non-finite losses or features")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "features": feature_tensor.contiguous(),
            "losses": loss_tensor.contiguous(),
            "targets": targets.contiguous(),
        },
        str(output / "oracle.safetensors"),
    )
    adapter_config = read_adapter_config(args.adapter)
    config = {
        "count": len(dataset),
        "depths": depths,
        "probe_depths": probe_depths,
        "margin": args.margin,
        "max_length": args.max_length,
        "seed": args.seed,
        "base_config": adapter_config["base_config"],
        "model": adapter_config["model"],
    }
    with (output / "oracle_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    print(f"Oracle saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
