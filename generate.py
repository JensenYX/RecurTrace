import argparse
import json

import torch
from transformers import GenerationConfig

from recurtrace import load_model, load_tokenizer
from recurtrace.data import SYSTEM_PROMPT, render_prompt


def main():
    parser = argparse.ArgumentParser(
        description="Generate an answer with fixed or adaptive loop depth."
    )
    parser.add_argument("--model", help="Public model name or a local base-model directory")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default=SYSTEM_PROMPT)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    args = parser.parse_args()
    if args.depth < 1 or args.max_new_tokens < 1 or not args.prompt.strip():
        parser.error("provide a nonempty prompt and positive depth and generation length")
    tokenizer = load_tokenizer(args.model, args.adapter)
    model = load_model(args.model, args.adapter, device=args.device, attention=args.attention)
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    prompt = render_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(args.device)
    if inputs.input_ids.shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
        parser.error("prompt and answer exceed the model's context length")
    depth = args.depth
    probabilities = None
    if args.adaptive:
        depth, probabilities = model.choose_depth(
            **inputs, min_depth=args.min_depth, max_depth=args.max_depth, threshold=args.threshold
        )
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=depth == 1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.inference_mode():
        output = model.generate(**inputs, loop_count=depth, generation_config=generation_config)
    answer = tokenizer.decode(
        output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    result = {"answer": answer, "depth": depth}
    if probabilities is not None:
        result["continue_probabilities"] = probabilities
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
