import torch
import torch.jit

from fme.core.typing_ import TensorDict


class DataShapesNotUniform(ValueError):
    """Indicates that a set of tensors do not all have the same shape."""

    pass


class Packer:
    """
    Responsible for packing tensors into a single tensor.
    """

    def __init__(self, names: list[str]):
        self.names = names

    def pack(self, tensors: TensorDict, axis=0) -> torch.Tensor:
        """
        Packs tensors into a single tensor, concatenated along a new axis.

        Args:
            tensors: Dict from names to tensors.
            axis: index for new concatenation axis.

        Raises:
            DataShapesNotUniform: when packed tensors do not all have the same shape.
        """
        # Normalize unexpected extra time dimension: if a tensor is 4D (B, T, H, W),
        # select the first time index to form (B, H, W). Inputs to the network are
        # single-timestep fields.
        processed = {}
        for name in self.names:
            t = tensors[name]
            # Normalize away singleton time dim and any extra singleton dims
            if t.dim() == 4:
                t = t.select(1, 0)
            elif t.dim() > 4:
                # iteratively squeeze singleton dims until <=4 then handle time
                while t.dim() > 4 and 1 in t.shape:
                    # squeeze the first singleton after batch dim
                    for d in range(1, t.dim() - 2):
                        if t.shape[d] == 1:
                            t = t.squeeze(d)
                            break
                    else:
                        break
                if t.dim() == 4:
                    t = t.select(1, 0)
            processed[name] = t

        shape = next(iter(processed.values())).shape
        mismatches = {n: v.shape for n, v in processed.items() if v.shape != shape}
        if mismatches:
            details = ", ".join([f"{n}:{s}" for n, s in mismatches.items()])
            raise DataShapesNotUniform(
                f"Cannot pack tensors of different shapes. Expected {shape}; mismatches: {details}"
            )
        return _pack(processed, self.names, axis=axis)

    def unpack(self, tensor: torch.Tensor, axis=0) -> TensorDict:
        return _unpack(tensor, self.names, axis=axis)


@torch.jit.script
def _pack(tensors: TensorDict, names: list[str], axis: int = 0) -> torch.Tensor:
    return torch.cat([tensors[n].unsqueeze(axis) for n in names], dim=axis)


@torch.jit.script
def _unpack(tensor: torch.Tensor, names: list[str], axis: int = 0) -> TensorDict:
    return {n: tensor.select(axis, index=i) for i, n in enumerate(names)}
