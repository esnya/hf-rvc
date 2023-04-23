import torch


def tensor_to_str(tensor) -> str:
    if isinstance(tensor, torch.Tensor):
        return f"{tensor.dtype}({list(tensor.shape)})"
    if isinstance(tensor, list):
        return (
            f"{type(tensor)}({len(tensor)})["
            + ", ".join([tensor_to_str(value) for value in tensor])
            + "]"
        )
    if isinstance(tensor, tuple):
        return (
            f"{type(tensor)}("
            + ", ".join([tensor_to_str(value) for value in tensor])
            + ")"
        )
    if isinstance(tensor, dict):
        return (
            f"{type(tensor)}("
            + ", ".join(
                [f"{key}: {tensor_to_str(value)}" for key, value in tensor.items()]
            )
            + ")"
        )
    return str(tensor)
