"""Frequency-position-preserving CNN and permutation-invariant wall supervision."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import pi, tau

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F

from .geometry import Shell, _normalized_wall_parameters

_KERNEL_SIZES = (9, 7, 5, 5)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    max_walls: int = 12
    conv_channels: tuple[int, int, int, int] = (16, 32, 48, 64)
    hidden_features: int = 128
    min_height_m: float = 0.001

    def __post_init__(self) -> None:
        if self.max_walls < 3 or self.hidden_features < 1:
            raise ValueError("the CNN requires at least three wall slots and one hidden feature")
        if len(self.conv_channels) != len(_KERNEL_SIZES) or min(self.conv_channels) < 1:
            raise ValueError("conv_channels must give four positive channel counts")
        if not 0 < self.min_height_m < float("inf"):
            raise ValueError("min_height_m must be finite and positive")


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Plane and elevation errors scale by radians/meters; BCE/CE use natural logs."""

    plane_weight: float = 1.0
    presence_weight: float = 1.0
    boundary_weight: float = 1.0
    height_weight: float = 1.0
    angle_scale_rad: float = 1.0
    offset_scale_m: float = 1.0
    validity_weight: float = 0.1
    min_adjacent_sine: float = 0.1
    min_edge_length_m: float = 0.05

    def __post_init__(self) -> None:
        if not (0 < self.angle_scale_rad < float("inf") and 0 < self.offset_scale_m < float("inf")):
            raise ValueError("angle and offset loss scales must be finite and positive")
        if not (0 < self.min_adjacent_sine <= 1 and 0 <= self.min_edge_length_m < float("inf")):
            raise ValueError("invalid boundary validity margins")


@dataclass(frozen=True, slots=True)
class Prediction:
    angles: Tensor
    offsets: Tensor
    presence_logits: Tensor
    floor: Tensor
    ceiling: Tensor
    successor_logits: Tensor

    @property
    def planes(self) -> Tensor:
        return torch.stack(
            (self.angles.cos(), self.angles.sin(), torch.zeros_like(self.angles), self.offsets), dim=-1
        )


class RoomCNN(nn.Module):
    """Map a one-channel log-magnitude response to wall and elevation predictions."""

    def __init__(self, n_bins: int, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        if n_bins < 1:
            raise ValueError("n_bins must be positive")
        self.n_bins = n_bins
        self.config = config
        blocks = []
        channels = 1
        positions = n_bins
        for width, kernel in zip(config.conv_channels, _KERNEL_SIZES, strict=True):
            blocks.extend((nn.Conv1d(channels, width, kernel, stride=2, padding=kernel // 2), nn.SiLU()))
            channels = width
            positions = (positions + 1) // 2
        self.encoder = nn.Sequential(
            *blocks, nn.Flatten(), nn.Linear(channels * positions, config.hidden_features), nn.SiLU()
        )
        slots = config.max_walls
        self.plane_head = nn.Linear(config.hidden_features, 2 * slots)
        self.presence_head = nn.Linear(config.hidden_features, slots)
        self.height_head = nn.Linear(config.hidden_features, 2)
        self.successor_head = nn.Linear(config.hidden_features, slots * slots)
        # Start wall angles spread around the circle instead of concentrated near zero.
        with torch.no_grad():
            self.plane_head.bias[:slots].copy_(torch.arange(slots) * (tau / slots) - pi)

    def forward(self, response: Tensor) -> Prediction:
        if response.ndim == 2:
            response = response.unsqueeze(1)
        if response.ndim != 3 or response.shape[1:] != (1, self.n_bins):
            raise ValueError(f"expected response shape [batch, {self.n_bins}] or [batch, 1, {self.n_bins}]")
        features = self.encoder(response)
        slots = self.config.max_walls
        angles, offsets = self.plane_head(features).split(slots, dim=1)
        floor, raw_height = self.height_head(features).unbind(dim=1)
        return Prediction(
            angles=angles,
            offsets=offsets,
            presence_logits=self.presence_head(features),
            floor=floor,
            ceiling=floor + F.softplus(raw_height) + self.config.min_height_m,
            successor_logits=self.successor_head(features).reshape(-1, slots, slots),
        )


def _assign(
    angles: NDArray, offsets: NDArray, target_angles: NDArray, target_offsets: NDArray, config: LossConfig
) -> tuple[NDArray, NDArray]:
    delta = angles[:, None] - target_angles
    angle_error = np.abs((delta + pi) % tau - pi) / config.angle_scale_rad
    offset_error = np.abs(offsets[:, None] - target_offsets) / config.offset_scale_m
    return linear_sum_assignment(angle_error + offset_error)


def match_planes(
    predicted_planes: ArrayLike, reference_planes: ArrayLike, config: LossConfig = LossConfig()
) -> tuple[NDArray, NDArray]:
    """Return rectangular Hungarian assignment rows and columns using signed outward planes."""
    def parameters(planes: ArrayLike) -> tuple[NDArray, NDArray]:
        array = np.asarray(planes, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError("planes must be an array with shape [wall slots, 4]")
        normals, offsets = _normalized_wall_parameters(array)
        return np.arctan2(normals[:, 1], normals[:, 0]), offsets

    angles, offsets = parameters(predicted_planes)
    target_angles, target_offsets = parameters(reference_planes)
    return _assign(angles, offsets, target_angles, target_offsets, config)


def _validity_loss(angles: Tensor, offsets: Tensor, config: LossConfig) -> Tensor:
    """In target order, favor well-conditioned corners and forward-directed finite edges."""
    nx, ny = angles.cos(), angles.sin()
    px, py = nx.roll(1), ny.roll(1)
    pd = offsets.roll(1)
    det = px * ny - py * nx
    parallel = F.relu(config.min_adjacent_sine - det.abs()).mean()

    # Use exact intersections outside the margin and a bounded, continuous reciprocal near zero.
    reciprocal = det / det.square().clamp_min(config.min_adjacent_sine**2)
    x = (py * offsets - ny * pd) * reciprocal
    y = (nx * pd - px * offsets) * reciprocal
    # CCW tangent for outward normal (nx, ny) is (-ny, nx), including at concave corners.
    edge_extent = (y.roll(-1) - y) * nx - (x.roll(-1) - x) * ny
    backwards = F.relu(config.min_edge_length_m - edge_extent).mean() / config.offset_scale_m
    return parallel + backwards


def supervised_loss(
    prediction: Prediction, shells: Sequence[Shell], config: LossConfig = LossConfig()
) -> tuple[Tensor, dict[str, Tensor], list[NDArray]]:
    """Compute supervised losses and return matched slot IDs in each reference's CCW order."""
    batch, n_slots = prediction.angles.shape
    if batch == 0 or len(shells) != batch:
        raise ValueError("expected a nonempty batch with one shell per prediction")
    # Only assignment is detached. Matched regression uses the original model outputs.
    detached = torch.stack((prediction.angles.detach(), prediction.offsets.detach()), dim=1)
    if detached.dtype == torch.bfloat16:
        detached = detached.float()
    parameters = detached.cpu().numpy()
    matches = []
    plane_terms, presence_terms, boundary_terms = [], [], []
    for i, shell in enumerate(shells):
        count = len(shell.footprint)
        if count > n_slots:
            raise ValueError(f"reference has {count} walls but the model only has {n_slots} slots")
        reference = shell.planes
        target_angles = np.arctan2(reference[:, 1], reference[:, 0])
        target_offsets = reference[:, 3]
        rows, cols = _assign(parameters[i, 0], parameters[i, 1], target_angles, target_offsets, config)
        match = np.empty(count, dtype=rows.dtype)
        match[cols] = rows
        matches.append(match)

        slots = torch.as_tensor(match, device=prediction.angles.device)
        angles = prediction.angles[i, slots]
        offsets = prediction.offsets[i, slots]
        delta = angles - angles.new_tensor(target_angles)
        angle_error = torch.atan2(delta.sin(), delta.cos()).abs().mean() / config.angle_scale_rad
        offset_error = (offsets - offsets.new_tensor(target_offsets)).abs().mean() / config.offset_scale_m
        plane_terms.append(angle_error + offset_error)

        logits = prediction.presence_logits[i]
        present = F.softplus(-logits[slots]).mean()
        if count < n_slots:
            absent = torch.ones(n_slots, device=slots.device, dtype=torch.bool)
            absent[slots] = False
            present = (present + F.softplus(logits[absent]).mean()) * 0.5
        presence_terms.append(present)

        successors = prediction.successor_logits[i, slots[:, None], slots[None, :]]
        targets = torch.arange(count, device=slots.device).roll(-1)
        boundary_terms.append(
            F.cross_entropy(successors, targets) + config.validity_weight * _validity_loss(angles, offsets, config)
        )

    target_elevations = prediction.floor.new_tensor([(shell.floor, shell.ceiling) for shell in shells])
    true_floor, true_ceiling = target_elevations.unbind(dim=1)
    floor_error = (prediction.floor - true_floor).abs()
    height_error = ((prediction.ceiling - prediction.floor) - (true_ceiling - true_floor)).abs()
    components = {
        "plane": torch.stack(plane_terms).mean(),
        "presence": torch.stack(presence_terms).mean(),
        "boundary": torch.stack(boundary_terms).mean(),
        "height": (floor_error + height_error).mean() / config.offset_scale_m,
    }
    total = (
        config.plane_weight * components["plane"]
        + config.presence_weight * components["presence"]
        + config.boundary_weight * components["boundary"]
        + config.height_weight * components["height"]
    )
    return total, components, matches
