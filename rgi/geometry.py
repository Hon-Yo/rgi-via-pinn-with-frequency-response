"""Finite vertical walls from a simple counterclockwise footprint or a plane cycle."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_softmax
from shapely.geometry import Polygon
from shapely.validation import explain_validity

_LENGTH_EPS_M = 1e-8
_ANGLE_EPS = 1e-7


def _cross(a: NDArray, b: NDArray) -> NDArray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _finite_number(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _normalized_wall_parameters(planes: NDArray) -> tuple[NDArray, NDArray]:
    """Return outward signed unit XY normals and offsets from an [N, 4] array."""
    if not np.isfinite(planes).all():
        raise ValueError("wall plane coefficients must be finite")
    scales = np.hypot(planes[:, 0], planes[:, 1])
    if not np.isfinite(scales).all() or np.any(scales <= _ANGLE_EPS):
        raise ValueError("wall plane has a near-zero or non-finite XY normal")
    if np.any(np.abs(planes[:, 2]) > _ANGLE_EPS * scales):
        raise ValueError("wall planes must be vertical (nz = 0)")
    offsets = planes[:, 3] / scales
    if not np.isfinite(offsets).all():
        raise ValueError("normalized plane offsets must be finite")
    return planes[:, :2] / scales[:, None], offsets


def _footprint(points: ArrayLike) -> tuple[NDArray, Polygon, NDArray]:
    try:
        xy = np.array(points, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("footprint must be an array of XY vertices") from exc
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3:
        raise ValueError("footprint must have at least three XY vertices")
    if not np.isfinite(xy).all():
        raise ValueError("footprint vertices must be finite")
    edges = np.roll(xy, -1, axis=0) - xy
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    if not np.isfinite(lengths).all() or np.any(lengths <= _LENGTH_EPS_M):
        raise ValueError("footprint has a degenerate edge; do not repeat the closing vertex")
    polygon = Polygon(xy)
    if not polygon.is_valid:
        raise ValueError(f"invalid footprint: {explain_validity(polygon)}")
    if not np.isfinite(polygon.area) or polygon.area <= 0:
        raise ValueError("footprint must have a finite positive area")
    if not polygon.exterior.is_ccw:
        raise ValueError("footprint vertices must be counterclockwise")
    tangents = edges / lengths[:, None]
    if np.any(np.abs(_cross(np.roll(tangents, 1, axis=0), tangents)) <= _ANGLE_EPS):
        raise ValueError("adjacent walls must not be parallel; remove redundant collinear vertices")
    xy.setflags(write=False)
    return xy, polygon, tangents


@dataclass(frozen=True, slots=True, eq=False)
class Pillar:
    """Interior rectangular footprint with optional absolute bottom/top elevations (m)."""

    footprint: NDArray
    floor: float | None = None
    ceiling: float | None = None
    _polygon: Polygon = field(init=False, repr=False)

    def __post_init__(self) -> None:
        xy, polygon, tangents = _footprint(self.footprint)
        if len(xy) != 4 or np.any(
            np.abs(np.sum(tangents * np.roll(tangents, -1, axis=0), axis=1)) > _ANGLE_EPS
        ):
            raise ValueError("pillar footprint must be rectangular")
        floor = None if self.floor is None else _finite_number(self.floor, "pillar floor")
        ceiling = None if self.ceiling is None else _finite_number(self.ceiling, "pillar ceiling")
        if floor is not None and ceiling is not None and ceiling <= floor:
            raise ValueError("pillar ceiling must be above its floor")
        object.__setattr__(self, "footprint", xy)
        object.__setattr__(self, "floor", floor)
        object.__setattr__(self, "ceiling", ceiling)
        object.__setattr__(self, "_polygon", polygon)

    def to_dict(self) -> dict:
        result = {"footprint": self.footprint.tolist()}
        if self.floor is not None:
            result["floor"] = self.floor
        if self.ceiling is not None:
            result["ceiling"] = self.ceiling
        return result

    @classmethod
    def from_dict(cls, obj: Mapping) -> "Pillar":
        if not isinstance(obj, Mapping) or "footprint" not in obj:
            raise ValueError("pillar annotation must contain a footprint")
        return cls(obj["footprint"], obj.get("floor"), obj.get("ceiling"))


@dataclass(frozen=True, slots=True, eq=False)
class Shell:
    """A CCW metric footprint (no repeated endpoint) and absolute floor/ceiling z."""

    footprint: NDArray
    floor: float
    ceiling: float
    pillars: Sequence[Pillar | Mapping] = ()
    planes: NDArray = field(init=False, repr=False)
    faces: NDArray = field(init=False, repr=False)
    _polygon: Polygon = field(init=False, repr=False)

    def __post_init__(self) -> None:
        xy, polygon, tangents = _footprint(self.footprint)
        floor = _finite_number(self.floor, "floor")
        ceiling = _finite_number(self.ceiling, "ceiling")
        if ceiling <= floor or not np.isfinite(ceiling - floor):
            raise ValueError("ceiling must be above floor with a finite room height")
        try:
            pillar_iter = iter(self.pillars)
        except TypeError as exc:
            raise ValueError("pillars must be a sequence of annotations") from exc
        pillars = tuple(p if isinstance(p, Pillar) else Pillar.from_dict(p) for p in pillar_iter)
        for pillar in pillars:
            if not polygon.contains_properly(pillar._polygon):
                raise ValueError("freestanding pillar footprint must lie strictly inside the shell")
            if any(z is not None and not floor <= z <= ceiling for z in (pillar.floor, pillar.ceiling)):
                raise ValueError("pillar elevation must lie inside the shell height")

        count = len(xy)
        planes = np.zeros((count, 4))
        planes[:, 0] = tangents[:, 1]
        planes[:, 1] = -tangents[:, 0]
        planes[:, 3] = -np.einsum("ij,ij->i", planes[:, :2], xy)
        if not np.isfinite(planes).all():
            raise ValueError("footprint produces non-finite plane coefficients")

        # Vertex winding has the same outward normal as the plane.
        faces = np.empty((count, 4, 3))
        next_xy = np.roll(xy, -1, axis=0)
        faces[:, 0, :2] = faces[:, 3, :2] = xy
        faces[:, 1, :2] = faces[:, 2, :2] = next_xy
        faces[:, :2, 2] = floor
        faces[:, 2:, 2] = ceiling
        planes.setflags(write=False)
        faces.setflags(write=False)
        object.__setattr__(self, "footprint", xy)
        object.__setattr__(self, "floor", floor)
        object.__setattr__(self, "ceiling", ceiling)
        object.__setattr__(self, "pillars", pillars)
        object.__setattr__(self, "planes", planes)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "_polygon", polygon)

    def to_dict(self) -> dict:
        count = len(self.footprint)
        return {
            "footprint": self.footprint.tolist(),
            "floor": self.floor,
            "ceiling": self.ceiling,
            "pillars": [pillar.to_dict() for pillar in self.pillars],
            "walls": [
                {
                    "id": i,
                    "predecessor": (i - 1) % count,
                    "successor": (i + 1) % count,
                    "plane": self.planes[i].tolist(),
                    "vertices": self.faces[i].tolist(),
                }
                for i in range(count)
            ],
        }

    @classmethod
    def from_dict(cls, obj: Mapping) -> "Shell":
        if not isinstance(obj, Mapping) or not {"footprint", "floor", "ceiling"} <= obj.keys():
            raise ValueError("shell annotation must contain footprint, floor, and ceiling")
        return cls(obj["footprint"], obj["floor"], obj["ceiling"], obj.get("pillars", ()))


def decode(planes: ArrayLike, order: Sequence[int], floor: float, ceiling: float) -> Shell:
    """Intersect consecutive selected planes along a predicted CCW cycle, or raise ValueError."""
    try:
        candidates = np.asarray(planes, dtype=np.float64)
        slots = np.asarray(order)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid plane coefficients or wall order") from exc
    if candidates.ndim != 2 or candidates.shape[1] != 4:
        raise ValueError("planes must be an array with shape [wall slots, 4]")
    if slots.ndim != 1 or len(slots) < 3 or not np.issubdtype(slots.dtype, np.integer):
        raise ValueError("wall order must list at least three integer slot indices")
    if np.any(slots < 0) or np.any(slots >= len(candidates)) or len(np.unique(slots)) != len(slots):
        raise ValueError("wall order contains duplicate or out-of-range slot indices")
    normals, offsets = _normalized_wall_parameters(candidates[slots])
    previous = np.roll(normals, 1, axis=0)
    previous_offsets = np.roll(offsets, 1)
    determinants = _cross(previous, normals)
    if np.any(np.abs(determinants) <= _ANGLE_EPS):
        raise ValueError("adjacent selected planes are parallel or nearly parallel")

    # Vertex i joins wall i - 1 to wall i; the footprint may be concave.
    vertices = np.column_stack((
        previous[:, 1] * offsets - normals[:, 1] * previous_offsets,
        normals[:, 0] * previous_offsets - previous[:, 0] * offsets,
    )) / determinants[:, None]
    shell = Shell(vertices, floor, ceiling)
    alignment = np.einsum("ij,ij->i", shell.planes[:, :2], normals)
    if np.any(alignment < 1.0 - _ANGLE_EPS):
        bad_slot = slots[np.flatnonzero(alignment < 1.0 - _ANGLE_EPS)[0]]
        raise ValueError(f"selected plane in slot {bad_slot} does not face outward on the decoded boundary")
    return shell


def select_order(
    presence_probabilities: ArrayLike, successor_logits: ArrayLike, threshold: float
) -> list[int]:
    """Greedily link selected slots by row log-probability without forming premature cycles."""
    try:
        probabilities = np.asarray(presence_probabilities, dtype=np.float64)
        logits = np.asarray(successor_logits, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("presence probabilities and successor logits must be numerical arrays") from exc
    cut = _finite_number(threshold, "presence threshold")
    if not 0 <= cut <= 1:
        raise ValueError("presence threshold must be in [0, 1]")
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("presence probabilities must be a finite vector in [0, 1]")
    if logits.shape != (len(probabilities), len(probabilities)) or not np.isfinite(logits).all():
        raise ValueError("successor logits must be a finite square matrix over wall slots")
    slots = np.flatnonzero(probabilities >= cut).tolist()
    count = len(slots)
    if count < 3:
        raise ValueError("fewer than three wall slots pass the presence threshold")

    selected_logits = logits[np.ix_(slots, slots)]
    np.fill_diagonal(selected_logits, -np.inf)
    scores = log_softmax(selected_logits, axis=1)
    ranked = sorted((-scores[i, j], i, j) for i in range(count) for j in range(count) if i != j)
    successor: dict[int, int] = {}
    has_predecessor: set[int] = set()
    for _, tail, head in ranked:
        if tail in successor or head in has_predecessor:
            continue
        end = head
        while end in successor:
            end = successor[end]
        if end == tail and len(successor) < count - 1:
            continue
        successor[tail] = head
        has_predecessor.add(head)
        if len(successor) == count:
            break
    # Before the final link, the available endpoints are those of disjoint paths.
    # An edge joining different paths can never become forbidden later in the greedy scan.
    order = [slots[0]]
    current = 0
    for _ in range(count - 1):
        current = successor[current]
        order.append(slots[current])
    return order


def footprint_iou(shell_a: Shell, shell_b: Shell) -> float:
    """Intersection-over-union of two outer footprints, excluding pillar annotations."""
    a, b = shell_a._polygon, shell_b._polygon
    intersection = a.intersection(b).area
    return float(intersection / (a.area + b.area - intersection))
