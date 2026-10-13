from alignment.cli import parse_config_args
from alignment.config import load_config
from alignment.files import is_run_complete
from alignment.lora import run_lora_dpo


def main():
    args = parse_config_args("TRL DPOTrainer with LoRA on the persona preference data.")
    config = load_config(args.config, args.overrides)

    if args.skip_if_complete and is_run_complete(config.run_dir):
        print(f"{config.run_dir} is already complete.")
        return

    summary = run_lora_dpo(config, resume=args.resume)
    print(summary)


if __name__ == "__main__":
    main()
