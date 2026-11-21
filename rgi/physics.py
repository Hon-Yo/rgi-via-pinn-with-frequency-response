"""Deterministic image-source simulation and a topology-conditional PyTorch VJP."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pyroomacoustics as pra
from shapely.geometry import Point, Polygon
import torch

from .audio import AudioConfig, Feature, spectral_magnitude, spectrum_from_rir
from .geometry import Shell, decode


@dataclass(frozen=True)
class SimulatorConfig:
    absorption: float = 0.25  # energy absorption, shared by all faces
    max_order: int = 3
    sound_speed: float = 343.0  # m/s
    angle_step: float = 0.001  # radians
    distance_step: float = 0.002  # meters
    step_retries: int = 3

    def __post_init__(self):
        if not (0 <= self.absorption < 1 and self.max_order >= 0 and self.sound_speed > 0):
            raise ValueError("invalid image-source settings")
        if not (self.angle_step > 0 and self.distance_step > 0 and self.step_retries >= 0):
            raise ValueError("invalid finite-difference settings")


@dataclass(frozen=True)
class Pose:
    receiver: tuple[float, float, float]
    sources: tuple[tuple[float, float, float], ...]
    gains: tuple[float, ...]

    def __post_init__(self):
        xyz, gains = np.asarray((self.receiver, *self.sources)), np.asarray(self.gains)
        if xyz.shape != (len(self.sources) + 1, 3) or not np.isfinite(xyz).all():
            raise ValueError("pose requires finite XYZ coordinates in meters")
        if not len(gains) == len(self.sources) >= 1 or not np.isfinite(gains).all() or not gains.any():
            raise ValueError("pose requires one signed coherent gain per speaker and at least one active speaker")
        if np.any(np.linalg.norm(xyz[1:][gains != 0] - xyz[0], axis=1) < 1e-4):
            raise ValueError("active source and receiver locations must be distinct")

    @classmethod
    def from_dict(cls, obj):
        return cls(tuple(obj["receiver"]), tuple(map(tuple, obj["sources"])), tuple(obj["gains"]))


def simulate_rir(shell: Shell, pose: Pose, audio: AudioConfig, settings: SimulatorConfig,
                 *, relative_arrival: bool = False) -> np.ndarray:
    polygon = Polygon(shell.footprint)
    active = [(src, gain) for src, gain in zip(pose.sources, pose.gains, strict=True) if gain != 0]
    for x, y, z in (pose.receiver, *(src for src, _ in active)):
        if not (shell.floor < z < shell.ceiling and polygon.contains(Point(x, y))):
            raise ValueError("registered source or receiver lies outside the shell")
    material = pra.Material(settings.absorption)
    room = pra.Room.from_corners(shell.footprint.T, fs=audio.sample_rate,
                                max_order=settings.max_order, materials=material,
                                air_absorption=False, ray_tracing=False, use_rand_ism=False)
    room.extrude(shell.ceiling - shell.floor, materials=material)
    room.set_sound_speed(settings.sound_speed)
    shift = np.array([0, 0, shell.floor])
    room.add_microphone(np.asarray(pose.receiver) - shift)
    for src, _ in active:
        room.add_source(np.asarray(src) - shift)
    room.compute_rir()

    # PRA's fractional delay FIR is centered half a filter later than physical time zero.
    # Keep the negative-time filter support when re-centering on an acoustic arrival.
    start = pra.constants.get("frac_delay_length") // 2
    if relative_arrival:
        receiver = np.asarray(pose.receiver)
        direct = min(round(np.linalg.norm(receiver - src) * audio.sample_rate / settings.sound_speed)
                     for src, _ in active)
        start += direct - audio.pre_arrival_samples
    rir = np.zeros(audio.rir_samples)
    for impulse, (_, gain) in zip(room.rir[0], active, strict=True):
        left, right = max(0, start), min(len(impulse), start + len(rir))
        if right > left:
            rir[left - start:right - start] += gain * impulse[left:right]
    if not np.isfinite(rir).all() or not np.any(rir):
        raise ValueError("simulator produced no finite nonzero RIR in the response window")
    return rir


def comparable_alignment(feature: Feature) -> bool:
    return feature.alignment_method in ("correlation", "marker")


def simulated_magnitude(shell, pose, feature, audio, settings):
    if feature.calibration not in ("none", "complex"):
        raise ValueError("reference-relative magnitudes have no equivalent simulator reference")
    if not comparable_alignment(feature):
        raise ValueError("simulator comparison requires documented marker or correlation alignment")
    rir = simulate_rir(shell, pose, audio, settings,
                       relative_arrival=feature.alignment_method == "correlation")
    return spectral_magnitude(spectrum_from_rir(rir, audio), feature.valid_mask, audio)


def spectral_error(simulated, measured, mask, delta_hz):
    if not np.any(mask):
        raise ValueError("cannot compare an empty valid frequency band")
    # Uniform grid integration: sum |dB error| * delta_Hz / sum valid delta_Hz.
    return float(np.sum(np.abs(simulated[mask] - measured[mask])) * delta_hz
                 / (np.count_nonzero(mask) * delta_hz))


def continuous_shell(vector, order):
    angles, offsets = vector[:-2].reshape(-1, 2).T
    planes = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles), offsets))
    return decode(planes, order, vector[-2], vector[-1])


class AcousticObjective:
    def __init__(self, n_walls, pose, feature, audio, simulator):
        self.order = tuple(range(n_walls))
        self.pose, self.feature, self.audio, self.simulator = pose, feature, audio, simulator
        grid = audio.frequency_hz
        self.delta_hz = grid[1] - grid[0]
        self.calls = 0
        self.retries = 0
        self.seconds = 0.0

    def __call__(self, vector):
        shell = continuous_shell(vector, self.order)
        self.calls += 1
        start = perf_counter()
        try:
            simulated = simulated_magnitude(shell, self.pose, self.feature, self.audio, self.simulator)
            return spectral_error(simulated, self.feature.magnitude_db, self.feature.valid_mask, self.delta_hz)
        finally:
            self.seconds += perf_counter() - start


def centered_gradient(vector, objective, settings):
    gradient = np.empty_like(vector)
    for i in range(len(vector)):
        step = settings.angle_step if i < len(vector) - 2 and i % 2 == 0 else settings.distance_step
        high, low = vector.copy(), vector.copy()
        for attempt in range(settings.step_retries + 1):
            high[i], low[i] = vector[i] + step, vector[i] - step
            try:
                gradient[i] = (objective(high) - objective(low)) / (2 * step)
                break
            except ValueError as exc:
                if attempt == settings.step_retries:
                    raise ValueError(f"no valid centered perturbation for parameter {i}: {exc}") from exc
                objective.retries += 1
                step *= 0.5
    return gradient


class FiniteDifferenceLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, geometry, objective, settings):
        vector = geometry.detach().cpu().double().numpy()
        value = objective(vector)
        # Compute eagerly so an invalid perturbation can omit the example before batch reduction.
        ctx.save_for_backward(geometry.new_tensor(centered_gradient(vector, objective, settings)))
        return geometry.new_tensor(value)

    @staticmethod
    def backward(ctx, upstream):
        (gradient,) = ctx.saved_tensors
        return upstream * gradient, None, None


def physics_loss(geometry, objective):
    return FiniteDifferenceLoss.apply(geometry, objective, objective.simulator)


def matched_geometry(prediction, row, slots):
    """Gather predicted continuous parameters in a fixed reference CCW wall order."""
    indices = torch.as_tensor(slots, device=prediction.angles.device)
    planes = torch.stack((prediction.angles[row, indices], prediction.offsets[row, indices]), dim=-1)
    return torch.cat((planes.flatten(), prediction.floor[row:row + 1], prediction.ceiling[row:row + 1]))


def shell_vector(shell):
    planes = shell.planes
    return np.r_[np.column_stack((np.arctan2(planes[:, 1], planes[:, 0]), planes[:, 3])).ravel(),
                 shell.floor, shell.ceiling]
