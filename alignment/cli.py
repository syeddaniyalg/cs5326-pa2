import argparse


def parse_config_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    parser.add_argument("--skip_if_complete", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest rolling training checkpoint when available.")
    return parser.parse_args()
