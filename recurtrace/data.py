import json
import math
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset


SYSTEM_PROMPT = "Answer with the final answer only."


def load_data(path, seed=42, limit=None):
    path = Path(path)
    if path.suffix == ".jsonl":
        entries = [{"path": path.name}]
    else:
        with path.open(encoding="utf-8") as handle:
            entries = json.load(handle)["files"]
    parts = []
    for entry in entries:
        weight = float(entry.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("data weights must be finite and nonnegative")
        if weight == 0:
            continue
        dataset = load_dataset("json", data_files=str(path.parent / entry["path"]), split="train")
        if "messages" not in dataset.column_names:
            raise ValueError(f"messages field missing in {entry['path']}")
        dataset = dataset.select_columns(["messages"])
        copies = math.floor(weight)
        parts.extend([dataset] * copies)
        count = round(len(dataset) * (weight - copies))
        if count:
            parts.append(dataset.shuffle(seed=seed).select(range(count)))
    if not parts:
        raise ValueError("no training examples were selected")
    dataset = concatenate_datasets(parts).shuffle(seed=seed)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        dataset = dataset.select(range(min(limit, len(dataset))))
    if not len(dataset):
        raise ValueError("the dataset is empty")
    return dataset


def messages_and_target(row):
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("each example needs a prompt and a final assistant answer")
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("message roles must be system, user, or assistant")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("message content must be a nonempty string")
    if messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
        raise ValueError("examples must end with a user prompt and an assistant answer")
    return messages[:-1], messages[-1]["content"]


def render_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def encode_example(tokenizer, row, max_length, include_eos=True):
    messages, target = messages_and_target(row)
    prompt = render_prompt(tokenizer, messages)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + target, add_special_tokens=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("the tokenizer merged the prompt boundary with the answer")
    answer_ids = full_ids[len(prompt_ids) :]
    if not answer_ids:
        raise ValueError("the answer must contain at least one token")
    if include_eos:
        answer_ids.append(tokenizer.eos_token_id)
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_length:
        raise ValueError(
            f"an example has {len(input_ids)} tokens; increase --max-length above {max_length}"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
        "prompt_mask": [1] * len(prompt_ids) + [0] * len(answer_ids),
    }


class AnswerCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        length = max(len(row["input_ids"]) for row in rows)
        padding = {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": 0,
            "labels": -100,
            "prompt_mask": 0,
        }
        return {
            key: torch.tensor(
                [list(row[key]) + [value] * (length - len(row[key])) for row in rows],
                dtype=torch.long,
            )
            for key, value in padding.items()
        }


def encode_batch(tokenizer, rows, max_length, device=None, include_eos=False):
    encoded = [encode_example(tokenizer, row, max_length, include_eos) for row in rows]
    batch = AnswerCollator(tokenizer)(encoded)
    return {key: value.to(device) for key, value in batch.items()}


def answer_nll(logits, labels):
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(-100)
    if not bool(valid.any(dim=1).all()):
        raise ValueError("each example must contain answer tokens")
    losses = torch.nn.functional.cross_entropy(
        shifted_logits.flatten(0, 1).float(),
        shifted_labels.flatten(),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_labels)
    return (losses * valid).sum(dim=1) / valid.sum(dim=1)
