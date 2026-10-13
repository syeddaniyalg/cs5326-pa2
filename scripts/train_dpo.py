from alignment.cli import parse_config_args
from alignment.config import load_config
from alignment.dpo import DPORunner
from alignment.files import is_run_complete


def main():
    args = parse_config_args("From-scratch Direct Preference Optimization.")
    config = load_config(args.config, args.overrides)

    if args.skip_if_complete and is_run_complete(config.run_dir):
        print(f"{config.run_dir} is already complete.")
        return

    summary = DPORunner(config, resume=args.resume).train()
    print(summary)


if __name__ == "__main__":
    main()
