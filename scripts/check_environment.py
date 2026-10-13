import argparse

import datasets
import torch
import transformers

from alignment.files import write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Print and save the compute environment.")
    parser.add_argument("--output_path", default="outputs/environment.json")
    parser.add_argument("--require_two_gpus", action="store_true")
    return parser.parse_args()


def describe_gpu(index):
    memory = torch.cuda.get_device_properties(index).total_memory
    return {"index": index, "name": torch.cuda.get_device_name(index), "memory_gib": round(memory / 2 ** 30, 1)}


def main():
    args = parse_args()
    gpus = [describe_gpu(index) for index in range(torch.cuda.device_count())]

    environment = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": len(gpus),
        "gpus": gpus,
    }
    write_json(args.output_path, environment)

    for key, value in environment.items():
        print(f"{key}: {value}")

    if args.require_two_gpus and len(gpus) < 2:
        raise SystemExit("Two GPUs are required: select the GPU T4 x2 accelerator.")


if __name__ == "__main__":
    main()
