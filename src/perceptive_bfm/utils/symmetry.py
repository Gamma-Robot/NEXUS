"""Left-right permutation and sign transforms for observations and actions."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class SymmetryTransform(nn.Module):
    def __init__(self, perm, signs, dim: int = -1):
        super().__init__()
        self.perm: torch.Tensor
        self.signs: torch.Tensor
        if not len(perm) == len(signs) > 0:
            raise ValueError("perm and signs must have the same length and be non-empty.")

        self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))
        self.register_buffer("signs", torch.as_tensor(signs, dtype=torch.float32))
        self.dim = dim

    def forward(self, x: torch.Tensor, sign: bool = True) -> torch.Tensor:
        # Normalize dim to positive index
        dim = self.dim if self.dim >= 0 else x.ndim + self.dim

        # Simple case: permute the last dimension
        if dim == x.ndim - 1:
            if sign:
                return x[..., self.perm] * self.signs
            else:
                return x[..., self.perm]

        # Complex case: flatten from dim to end, permute, reshape back
        # Save original shape
        original_shape = list(x.shape)

        # Flatten from dim to end
        flat_size = 1
        for i in range(dim, x.ndim):
            flat_size *= original_shape[i]

        # Reshape to (everything before dim, flat_size)
        x = x.reshape(original_shape[:dim] + [flat_size])

        # Apply permutation to last dimension
        x = x[..., self.perm]

        # Apply signs
        if sign:
            x = x * self.signs

        # Reshape back to original shape
        x = x.reshape(original_shape)

        return x

    def repeat(self, n: int) -> SymmetryTransform:
        return SymmetryTransform.cat([self] * n)

    @staticmethod
    def cat(transforms: Sequence[SymmetryTransform]) -> SymmetryTransform:
        if not all(isinstance(t, SymmetryTransform) for t in transforms):
            raise ValueError("All transforms must be SymmetryTransform instances.")
        perm = []
        signs = []
        num = 0
        for t in transforms:
            perm.append(t.perm + num)
            signs.append(t.signs)
            num += t.perm.shape[0]
        # All transforms must have the same dim
        dims = [t.dim for t in transforms]
        if not all(d == dims[0] for d in dims):
            raise ValueError("All transforms must have the same dim parameter.")
        return SymmetryTransform(torch.cat(perm), torch.cat(signs), dim=dims[0])


def mirrored(symmetry_mapping: dict) -> dict:
    """Add the reverse direction of each mapping entry in place, return the same dict."""
    reverse = {}
    for k, v in symmetry_mapping.items():
        if isinstance(v, tuple):  # joint-space symmetry: (sign, other_name)
            reverse[v[1]] = (v[0], k)
        elif isinstance(v, str):  # cartesian-space symmetry: other_name
            reverse[v] = k
        else:
            raise ValueError(f"Invalid symmetry mapping: ({k}, {v})")
    symmetry_mapping.update(reverse)
    return symmetry_mapping


def joint_space_symmetry(
    joint_names: Sequence[str], symmetry_mapping: dict[str, tuple[float, str]]
) -> SymmetryTransform:
    """Permutation that transforms a vector of joint quantities (position order:
    ``joint_names``) into its left-right mirrored counterpart."""
    if not len(symmetry_mapping) == len(joint_names):
        diff_missing = set(joint_names) - set(symmetry_mapping.keys())
        diff_extra = set(symmetry_mapping.keys()) - set(joint_names)
        raise ValueError(
            "Joint symmetry mapping must contain all joint names\n"
            f"\tAll Joints - Specified: {diff_missing}\n"
            f"\tSpecified - All Joints: {diff_extra}"
        )

    ids = torch.zeros(len(joint_names), dtype=torch.long)
    signs = torch.zeros(len(joint_names), dtype=torch.float32)
    for i, this_joint_name in enumerate(joint_names):
        sign, other_joint_name = symmetry_mapping[this_joint_name]
        ids[i] = joint_names.index(other_joint_name)
        signs[i] = sign
    return SymmetryTransform(ids, signs)


def cartesian_space_symmetry(
    names: Sequence[str],
    symmetry_mapping: dict[str, str],
    sign: Sequence[float] = (1, -1, 1),
) -> SymmetryTransform:
    """Permutation that transforms a per-name block vector (each name contributing
    ``len(sign)`` consecutive elements, in ``names`` order) into its left-right
    mirrored counterpart."""
    if not len(symmetry_mapping) >= len(names):
        raise ValueError(
            "Spatial symmetry mapping must contain all requested names\n"
            f"\tRequested - Specified: {set(names) - set(symmetry_mapping.keys())}"
        )

    ids = torch.zeros(len(names), len(sign), dtype=torch.long)
    signs = torch.zeros(len(names), len(sign), dtype=torch.float32)
    for i, this_name in enumerate(names):
        other_name = symmetry_mapping[this_name]
        ids[i] = names.index(other_name) * len(sign) + torch.arange(len(sign))
        signs[i] = torch.tensor(sign, dtype=torch.float32)
    return SymmetryTransform(ids.flatten(), signs.flatten())


def grid_symmetry(nx: int, ny: int, dim: int = -1) -> SymmetryTransform:
    """Left-right mirror for a flattened ``(ny, nx)`` grid of scalar samples (e.g. a
    height scan / elevation map z-channel). Matches mjlab ``GridPatternCfg.generate_rays``
    ray ordering: ``meshgrid(x, y, indexing="xy")`` flattened row-major -> ``(ny, nx)``,
    ``nx`` = grid length (forward/x), ``ny`` = grid width (lateral/y).

    Mirroring left/right flips the lateral (``y``/``ny``) axis and keeps forward
    (``x``/``nx``) order; the sampled values themselves are scalar (no vector
    components), so no sign flip -- permutation only.

    ``dim`` selects which axis of the tensor this transform is applied to when composed
    via ``SymmetryTransform.cat``/``.repeat`` (default -1, the trailing axis, matching
    a flattened ``(ny * nx,)`` vector -- pass a different ``dim`` for e.g. a
    ``(..., ny, nx)`` grid tensor where the axis to flip is not the last one).
    """
    perm = torch.arange(ny * nx).reshape(ny, nx).flip(0).flatten()
    return SymmetryTransform(perm, torch.ones(ny * nx), dim=dim)
