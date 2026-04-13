import argparse
import os
import sys

from dotenv import load_dotenv

from train import train

load_dotenv()


def _load_yaml(path: str) -> dict:
    import yaml  # lazy import so non-YAML runs don't require PyYAML
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return cfg


def add_train_args(p):
    # --config is parsed first; its values become argparse defaults,
    # so anything also passed on the CLI transparently overrides YAML.
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config file. CLI flags override YAML values.")

    # Data
    p.add_argument("--data_root", type=str, default="lab3/common_voice_mn",
                   help="Local path to common_voice_mn dir containing validated.tsv and clips/")
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--test_ratio", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_input_length", type=float, default=20.0,
                   help="Max audio length in seconds (Whisper hard limit is 30).")

    # Model
    p.add_argument("--model_name_or_path", type=str, default="openai/whisper-medium")
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--no_lora", dest="use_lora", action="store_false")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # Optim
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--mixed_precision", type=str, default="bf16",
                   choices=["no", "fp16", "bf16"],
                   help="bf16 recommended on RTX 5080 (Blackwell).")

    # Eval
    p.add_argument("--eval_num_beams", type=int, default=1)

    # Early stopping / misc
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    p.add_argument("--output_dir", type=str, default="./whisper-medium-mn-lora")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug_subset_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--auth_token", type=str, default=os.getenv("HF_TOKEN"))


def add_eval_args(p):
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data_root", type=str, default="lab3/common_voice_mn")
    p.add_argument("--model_dir", type=str, default="./whisper-medium-mn-lora")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_input_length", type=float, default=20.0)
    p.add_argument("--num_beams", type=int, default=5)
    # Must mirror the same split config used at train time so the held-out
    # test rows are the same ones the model never saw.
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--test_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)


def _apply_config(parser: argparse.ArgumentParser, argv: list) -> argparse.Namespace:
    """Parse once to find --config, reset defaults from YAML, then parse again.

    This gives the precedence order: CLI > YAML > argparse defaults.
    """
    args, _ = parser.parse_known_args(argv)
    cfg_path = getattr(args, "config", None)
    if cfg_path:
        cfg = _load_yaml(cfg_path)
        valid = {a.dest for a in parser._actions}
        unknown = set(cfg) - valid
        if unknown:
            raise ValueError(f"Unknown keys in {cfg_path}: {sorted(unknown)}")
        parser.set_defaults(**cfg)
    return parser.parse_args(argv)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper-medium on Common Voice Mongolian (local) with LoRA."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Fine-tune Whisper-medium")
    add_train_args(train_parser)

    eval_parser = sub.add_parser("eval", help="Evaluate checkpoint on the test split")
    add_eval_args(eval_parser)

    # Two-pass parse so YAML can feed defaults to the active subcommand.
    top_args, _ = parser.parse_known_args()
    if top_args.command == "train":
        args = _apply_config(train_parser, sys.argv[2:])
        args.command = "train"
        train(args)
    elif top_args.command == "eval":
        from eval import evaluate_checkpoint
        args = _apply_config(eval_parser, sys.argv[2:])
        evaluate_checkpoint(
            model_dir=args.model_dir,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_input_length=args.max_input_length,
            num_beams=args.num_beams,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
