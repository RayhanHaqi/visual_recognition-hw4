import argparse
from pathlib import Path

import torch


def _normalize_key(key):
    if key.startswith("module.net."):
        return key[len("module.net."):]
    if key.startswith("net."):
        return key[len("net."):]
    if key.startswith("module."):
        return key[len("module."):]
    return key


def _load_model_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    return {_normalize_key(key): value for key, value in state_dict.items()}


def average_checkpoints(checkpoint_paths, output_path):
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required")

    states = [_load_model_state(path) for path in checkpoint_paths]
    keys = set(states[0])
    for state in states[1:]:
        if set(state) != keys:
            raise ValueError("Checkpoint keys do not match")

    averaged = {}
    for key in sorted(keys):
        tensors = [state[key] for state in states]
        first = tensors[0]
        if not all(tensor.shape == first.shape for tensor in tensors):
            raise ValueError(f"Checkpoint tensor shapes do not match for {key}")

        if torch.is_floating_point(first):
            value = sum(tensor.float() for tensor in tensors) / len(tensors)
            averaged[key] = value.to(dtype=first.dtype)
        else:
            averaged[key] = first.clone()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged, output_path)
    print(f"Averaged {len(states)} checkpoint(s) into {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    average_checkpoints([Path(path) for path in args.checkpoints], args.output)


if __name__ == "__main__":
    main()
