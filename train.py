import argparse

import torch
from transformers import Trainer, TrainingArguments, set_seed

from recurtrace import load_model, load_tokenizer, save_adapter
from recurtrace.data import AnswerCollator, encode_example, load_data
from recurtrace.loading import BASE_MODEL


def main():
    parser = argparse.ArgumentParser(
        description="Train RecurTrace loop memory on direct-answer data."
    )
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--data", required=True, help="JSONL file or a JSON manifest")
    parser.add_argument("--output", default="outputs/stage1")
    parser.add_argument("--loop-start", type=int)
    parser.add_argument("--loop-layers", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--mean-depth", type=float, default=4.0)
    parser.add_argument("--depth-sigma", type=float, default=0.5)
    parser.add_argument("--memory-window", type=int, default=3)
    parser.add_argument("--epochs", type=float, default=15)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    args = parser.parse_args()
    if args.batch_size < 1 or args.gradient_accumulation < 1 or args.max_length < 2:
        parser.error("batch size, accumulation and sequence length must be positive")
    if args.max_steps == 0 or args.max_steps < -1 or args.epochs <= 0:
        parser.error("provide positive --epochs and either positive --max-steps or -1")
    if args.learning_rate <= 0 or not 0 <= args.warmup_ratio <= 1 or args.weight_decay < 0:
        parser.error("invalid optimizer settings")
    use_cpu = args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    bf16 = not use_cpu and torch.cuda.is_bf16_supported()
    set_seed(args.seed)
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        bf16=bf16,
        use_cpu=use_cpu,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=True,
        max_grad_norm=1.0,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_pin_memory=not use_cpu,
        remove_unused_columns=False,
    )
    tokenizer = load_tokenizer(args.model)
    model = load_model(
        args.model,
        dtype=torch.bfloat16 if bf16 else torch.float32,
        attention=args.attention,
        loop_start_layer=args.loop_start,
        loop_num_layers=args.loop_layers,
        loop_train_max=args.max_depth,
        loop_train_mean=args.mean_depth,
        loop_train_sigma=args.depth_sigma,
        loop_memory_window=args.memory_window,
        loop_seed=args.seed,
    )
    model.config.use_cache = False
    model.model.loop_attn.float()
    model.model.loop_input_injection.float()
    with training_args.main_process_first():
        dataset = load_data(args.data, seed=args.seed, limit=args.limit)
        dataset = dataset.map(
            lambda row: encode_example(tokenizer, row, args.max_length),
            remove_columns=dataset.column_names,
            desc="Tokenizing answers",
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=AnswerCollator(tokenizer),
        processing_class=tokenizer,
    )
    if trainer.is_world_process_zero():
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        print(f"Training {trainable:,} parameters on {len(dataset):,} examples.", flush=True)
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        save_adapter(trainer.accelerator.unwrap_model(trainer.model), args.output, tokenizer)
        print(f"Adapter saved to {args.output}", flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
