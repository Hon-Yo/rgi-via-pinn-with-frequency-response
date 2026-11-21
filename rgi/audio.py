"""WAV deconvolution, shared acoustic features and interchange formats."""

from dataclasses import dataclass
from math import gcd
from numbers import Integral
from pathlib import Path
import json

import numpy as np
from scipy import fft, io, ndimage, signal


_CALIBRATIONS = frozenset({"none", "complex", "magnitude", "covered"})
_VECTOR_FIELDS = (
    "frequency_hz", "magnitude_db", "valid_mask", "transfer_real", "transfer_imag",
)


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 24000
    f_min: float = 100.0
    f_max: float = 8000.0
    frequency_step_hz: float = 10.0
    rir_seconds: float = 0.5
    pre_arrival_seconds: float = 0.005
    taper_seconds: float = 0.025
    magnitude_floor: float = 1e-8
    regularization: float = 1e-8
    excitation_floor_db: float = -50.0
    reference_floor_db: float = -40.0
    envelope_smoothing_hz: float | None = None
    remove_scalar_level: bool = False
    clipping_peak: float = 0.999
    min_signal_rms: float = 1e-7

    def __post_init__(self) -> None:
        if not _is_integer(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        finite = (
            self.f_min, self.f_max, self.frequency_step_hz, self.rir_seconds,
            self.pre_arrival_seconds, self.taper_seconds, self.magnitude_floor,
            self.regularization, self.excitation_floor_db, self.reference_floor_db,
            self.clipping_peak, self.min_signal_rms,
        )
        if not np.isfinite(finite).all():
            raise ValueError("audio settings must be finite")
        if not (0 <= self.f_min < self.f_max <= self.sample_rate / 2):
            raise ValueError("frequency band must lie in [0, Nyquist] with f_min < f_max")
        if self.frequency_step_hz <= 0:
            raise ValueError("frequency_step_hz must be positive")
        steps = (self.f_max - self.f_min) / self.frequency_step_hz
        if not np.isfinite(steps) or steps < 0.5 or abs(steps - round(steps)) > 1e-8:
            raise ValueError("frequency_step_hz must reach f_max exactly from f_min")
        if self.rir_samples <= 0:
            raise ValueError("RIR duration must contain at least one sample")
        if not (0 <= self.pre_arrival_seconds < self.rir_seconds
                and self.pre_arrival_samples < self.rir_samples):
            raise ValueError("pre-arrival margin must be shorter than the RIR")
        if not (0 <= self.taper_seconds < self.rir_seconds
                and self.taper_samples < self.rir_samples):
            raise ValueError("tail taper must be shorter than the RIR")
        if self.magnitude_floor <= 0 or not 0 < self.regularization < 1:
            raise ValueError("magnitude floor must be positive and regularization in (0, 1)")
        if self.excitation_floor_db > 0 or self.reference_floor_db > 0:
            raise ValueError("relative excitation and reference floors must be nonpositive dB")
        if self.envelope_smoothing_hz is not None and (
            not np.isfinite(self.envelope_smoothing_hz) or self.envelope_smoothing_hz <= 0
        ):
            raise ValueError("envelope_smoothing_hz must be positive when set")
        if not (0 < self.min_signal_rms < self.clipping_peak <= 1):
            raise ValueError("require 0 < min_signal_rms < clipping_peak <= 1")

    @property
    def frequency_hz(self) -> np.ndarray:
        steps = round((self.f_max - self.f_min) / self.frequency_step_hz)
        return np.linspace(self.f_min, self.f_max, steps + 1)

    @property
    def rir_samples(self) -> int:
        return round(self.rir_seconds * self.sample_rate)

    @property
    def pre_arrival_samples(self) -> int:
        return round(self.pre_arrival_seconds * self.sample_rate)

    @property
    def taper_samples(self) -> int:
        return round(self.taper_seconds * self.sample_rate)


@dataclass
class Feature:
    frequency_hz: np.ndarray
    magnitude_db: np.ndarray
    valid_mask: np.ndarray
    transfer_real: np.ndarray
    transfer_imag: np.ndarray
    calibration: str = "none"
    impulse_response: np.ndarray | None = None
    system_transfer_real: np.ndarray | None = None
    system_transfer_imag: np.ndarray | None = None
    alignment_method: str | None = None
    timing_reference_seconds: float | None = None
    rir_start_seconds: float | None = None
    diagnostics: dict | None = None

    @property
    def transfer(self) -> np.ndarray:
        return self.transfer_real + 1j * self.transfer_imag


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _vector(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    return array


def _real_vector(value: object, name: str) -> np.ndarray:
    array = _vector(value, name)
    if array.dtype.kind not in "uif" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} values must fit in float64")
    return array


def _mask_vector(value: object) -> np.ndarray:
    array = _vector(value, "valid_mask")
    if array.dtype.kind not in "buif" or not np.all((array == 0) | (array == 1)):
        raise ValueError("valid_mask must contain only boolean values or 0/1")
    mask = array.astype(bool, copy=False)
    if not mask.any():
        raise ValueError("empty valid frequency band")
    return mask


def _transfer_vector(value: object, size: int) -> np.ndarray:
    array = _vector(value, "transfer")
    if array.size != size or array.dtype.kind not in "uifc":
        raise ValueError("transfer must contain finite numbers on the configured frequency grid")
    array = array.astype(np.complex128 if array.dtype.kind == "c" else np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("transfer must contain finite numbers on the configured frequency grid")
    return array


def _common_grid(value: object, expected: np.ndarray) -> np.ndarray:
    grid = _real_vector(value, "frequency_hz")
    step = expected[1] - expected[0]
    # Accept the rounding of a MATLAB single-precision vector, not a shifted grid.
    tolerance = min(step * 1e-3, np.finfo(np.float32).eps * max(1.0, expected[-1]))
    if grid.shape != expected.shape or not np.allclose(grid, expected, atol=tolerance, rtol=0):
        raise ValueError("frequency_hz does not match the configured uniform frequency grid")
    return expected


def spectrum_from_rir(rir: np.ndarray, config: AudioConfig) -> np.ndarray:
    """Crop/pad and taper a real RIR whose sample zero is the chosen window origin.

    The trailing raised cosine falls to zero at the last retained sample. The
    zero-padded transform is interpolated in the complex plane (no phase unwrap).
    Padding improves interpolation, not true resolution.
    """
    raw = _real_vector(rir, "rir")
    window = np.zeros(config.rir_samples)
    count = min(raw.size, window.size)
    window[:count] = raw[:count]
    taper = config.taper_samples
    if taper:
        phase = np.arange(1, taper + 1) * (np.pi / taper)
        window[-taper:] *= 0.5 * (1.0 + np.cos(phase))
    nfft = fft.next_fast_len(8 * config.rir_samples, real=True)
    hz = fft.rfftfreq(nfft, 1 / config.sample_rate)
    return np.interp(config.frequency_hz, hz, fft.rfft(window, nfft))


def _log_magnitude(transfer: np.ndarray, config: AudioConfig) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(transfer), config.magnitude_floor))


def _transform_magnitude(db: np.ndarray, mask: np.ndarray, config: AudioConfig) -> np.ndarray:
    if config.envelope_smoothing_hz is not None:
        sigma = config.envelope_smoothing_hz / config.frequency_step_hz
        numerator = ndimage.gaussian_filter1d(np.where(mask, db, 0.0), sigma, mode="constant")
        weight = ndimage.gaussian_filter1d(mask.astype(float), sigma, mode="constant")
        db = db - np.divide(numerator, weight, out=np.zeros_like(db), where=weight > 0)
    if config.remove_scalar_level:
        db = db - db[mask].mean()
    return db


def spectral_magnitude(transfer: np.ndarray, mask: np.ndarray, config: AudioConfig) -> np.ndarray:
    """Compute the unstandardized dB feature, applying configured transforms on valid bins."""
    grid = config.frequency_hz
    transfer = _transfer_vector(transfer, grid.size)
    valid = _mask_vector(mask)
    if valid.shape != grid.shape:
        raise ValueError("valid_mask must match the configured frequency grid")
    return _transform_magnitude(_log_magnitude(transfer, config), valid, config)


def _read_channel(
    path: str | Path, channel: int, expected_rate: int | None, config: AudioConfig,
) -> tuple[np.ndarray, dict]:
    if not _is_integer(channel) or channel < 0:
        raise ValueError(f"{path}: channel must be a nonnegative integer")
    if expected_rate is not None and (not _is_integer(expected_rate) or expected_rate <= 0):
        raise ValueError(f"{path}: expected sample rate must be a positive integer")
    rate, data = io.wavfile.read(path)
    if rate <= 0 or (expected_rate is not None and rate != expected_rate):
        raise ValueError(f"{path}: WAV sample rate {rate} does not match expected {expected_rate}")
    if data.ndim not in (1, 2) or data.size == 0:
        raise ValueError(f"{path}: WAV must contain nonempty mono or multichannel samples")
    channels = data.shape[1] if data.ndim == 2 else 1
    if channel >= channels:
        raise ValueError(f"{path}: channel {channel} unavailable in WAV with {channels} channel(s)")
    selected = data[:, channel] if data.ndim == 2 else data
    at_positive_rail = False
    if selected.dtype.kind == "f" and selected.dtype.itemsize in (4, 8):
        samples = selected.astype(np.float64, copy=False)
    elif selected.dtype.kind == "i":
        # SciPy left-justifies packed PCM in its output dtype (e.g. 24 in 32 bits).
        samples = np.ldexp(selected.astype(np.float64), 1 - 8 * selected.dtype.itemsize)
        at_positive_rail = bool(selected.max() == np.iinfo(selected.dtype).max)
    elif selected.dtype == np.dtype(np.uint8):
        samples = (selected.astype(np.float64) - 128) / 128
        at_positive_rail = bool(selected.max() == 255)
    else:
        raise ValueError(f"{path}: expected uncompressed PCM or floating-point WAV")
    if not np.isfinite(samples).all():
        raise ValueError(f"{path}: selected channel contains nonfinite samples")
    peak = float(np.max(np.abs(samples)))
    if peak >= config.clipping_peak or at_positive_rail:
        raise ValueError(f"{path}: selected channel is clipped or near full scale (peak {peak:.4g})")
    rms = float(samples.std())
    if rms < config.min_signal_rms:
        raise ValueError(f"{path}: selected channel is silent or constant (AC RMS {rms:.4g})")
    metadata = {
        "sample_rate": int(rate), "channel": int(channel), "channels": channels,
        "samples": int(samples.size), "duration_seconds": samples.size / rate,
        "wav_sample_dtype": selected.dtype.str, "peak": peak, "ac_rms": rms,
    }
    if rate != config.sample_rate:
        divisor = gcd(rate, config.sample_rate)
        samples = signal.resample_poly(samples, config.sample_rate // divisor, rate // divisor)
    metadata["resampled_samples"] = int(samples.size)
    return samples, metadata


def _apply_calibration(
    system: np.ndarray, mask: np.ndarray, calibration: str, reference: Feature | None,
    grid: np.ndarray, config: AudioConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    if calibration == "none":
        if reference is not None:
            raise ValueError("a calibration reference requires a calibration mode other than 'none'")
        return system, _log_magnitude(system, config), mask, None
    if reference is None:
        raise ValueError(f"{calibration} calibration requires a reference feature")
    _common_grid(reference.frequency_hz, grid)
    real = _real_vector(reference.transfer_real, "reference transfer_real")
    imag = _real_vector(reference.transfer_imag, "reference transfer_imag")
    reference_mask = _mask_vector(reference.valid_mask)
    if any(a.shape != grid.shape for a in (real, imag, reference_mask)):
        raise ValueError("reference transfer and valid_mask must match the frequency grid")
    device = real + 1j * imag
    amplitude = np.abs(device)
    peak = float(amplitude[reference_mask].max())
    threshold = max(config.magnitude_floor, peak * (10 ** (config.reference_floor_db / 20)))
    valid = mask & reference_mask & (amplitude >= threshold)
    if not valid.any():
        raise ValueError("empty valid frequency band after calibration reference masking")
    if calibration == "complex":
        # Normalize the device first so the ridge is relative to its peak power.
        unit = np.divide(device, peak, out=np.zeros_like(device), where=valid)
        corrected = (system / peak) * unit.conj() / (np.abs(unit) ** 2 + config.regularization)
        return corrected, _log_magnitude(corrected, config), valid, threshold
    # In these modes the complex response still represents the measured system.
    db = _log_magnitude(system, config) - _log_magnitude(device, config)
    return system, db, valid, threshold


def process_wavs(
    excitation_path: str | Path, recording_path: str | Path, config: AudioConfig, *,
    excitation_channel: int = 0, recording_channel: int = 0,
    excitation_rate: int | None = None, recording_rate: int | None = None,
    sweep_start_seconds: float | None = None, calibration: str = "none",
    reference: Feature | None = None,
) -> Feature:
    """Identify one WAV pair at the configured rate and response duration.

    A marker gives the time in the microphone WAV corresponding to *sample zero
    of the excitation file*, not the first non-silent sample. The RIR starts at
    the marker, retaining acoustic delay. Without a marker the strongest absolute
    correlation at a nonnegative lag is the acoustic reference (not necessarily
    a direct arrival); the RIR begins a fixed pre-arrival margin before it. If
    this crosses recording sample zero, those leading RIR samples are zero-filled
    and counted in diagnostics. Transfer phase is relative to the RIR start.
    """
    if calibration not in _CALIBRATIONS:
        raise ValueError(f"unknown calibration mode {calibration!r}")
    if sweep_start_seconds is not None and (
        not np.isfinite(sweep_start_seconds) or sweep_start_seconds < 0
    ):
        raise ValueError("sweep_start_seconds must be finite and nonnegative")
    x, x_info = _read_channel(excitation_path, excitation_channel, excitation_rate, config)
    y, y_info = _read_channel(recording_path, recording_channel, recording_rate, config)
    if y.size < x.size:
        raise ValueError("recording does not cover the complete excitation and required response tail")
    # The full linear cross-correlation then fits, with positive and negative
    # lags disjoint. An additional coverage check ensures the desired RIR is
    # observable even if the recording is shorter than the supported convolution.
    nfft = fft.next_fast_len(max(x.size + y.size - 1, x.size + config.rir_samples - 1), real=True)
    x_spectrum = fft.rfft(x, nfft)
    power = np.abs(x_spectrum) ** 2
    peak_power = float(power.max())
    cross = fft.rfft(y, nfft) * x_spectrum.conj()
    if sweep_start_seconds is None:
        correlation = fft.irfft(cross, nfft)
        timing_reference = int(np.argmax(np.abs(correlation[:y.size])))
        start = timing_reference - config.pre_arrival_samples
        method = "correlation"
    else:
        timing_reference = round(sweep_start_seconds * config.sample_rate)
        start = timing_reference
        method = "marker"
    stop = start + config.rir_samples
    required_samples = x.size + stop - 1
    if required_samples > y.size:
        missing = (required_samples - y.size) / config.sample_rate
        raise ValueError(f"recording does not cover the required response tail ({missing:.6f} s short)")

    # Use only the samples before the RIR origin as a conservative preroll.
    preroll = y[:max(0, start)]
    y_info["pre_sweep_samples"] = int(preroll.size)
    y_info["pre_sweep_ac_rms"] = float(preroll.std()) if preroll.size else None

    response = fft.irfft(cross / (power + config.regularization * peak_power), nfft)
    left_padding = max(0, -start)
    rir = np.zeros(config.rir_samples)
    rir[left_padding:] = response[max(0, start):stop]
    grid = config.frequency_hz
    excitation_power = np.interp(grid, fft.rfftfreq(nfft, 1 / config.sample_rate), power)
    mask = excitation_power >= peak_power * (10 ** (config.excitation_floor_db / 10))
    source_nyquist = min(x_info["sample_rate"], y_info["sample_rate"]) / 2
    mask &= grid <= source_nyquist
    if not mask.any():
        raise ValueError("empty valid frequency band for this excitation")
    system = spectrum_from_rir(rir, config)
    transfer, db, valid, reference_threshold = _apply_calibration(
        system, mask, calibration, reference, grid, config,
    )
    return Feature(
        grid, _transform_magnitude(db, valid, config), valid, transfer.real, transfer.imag,
        calibration=calibration, impulse_response=rir,
        system_transfer_real=system.real if calibration == "complex" else None,
        system_transfer_imag=system.imag if calibration == "complex" else None,
        alignment_method=method, timing_reference_seconds=timing_reference / config.sample_rate,
        rir_start_seconds=start / config.sample_rate,
        diagnostics={
            "excitation": x_info, "recording": y_info, "sample_rate": int(config.sample_rate),
            "deconvolution_fft_samples": int(nfft), "rir_samples": config.rir_samples,
            "rir_duration_seconds": config.rir_samples / config.sample_rate,
            "taper_samples": config.taper_samples, "pre_arrival_zero_padding_samples": left_padding,
            "f_min": config.f_min, "f_max": config.f_max, "frequency_step_hz": config.frequency_step_hz,
            "excitation_floor_db": config.excitation_floor_db,
            "excitation_valid_bins": int(mask.sum()), "valid_bins": int(valid.sum()),
            "reference_magnitude_threshold": reference_threshold,
            "magnitude_floor": config.magnitude_floor, "regularization": config.regularization,
            "envelope_smoothing_hz": config.envelope_smoothing_hz,
            "remove_scalar_level": config.remove_scalar_level,
        },
    )


def _text_scalar(value: object, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a text scalar")
    item = array.item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _finite_scalar(value: object, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "uif" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite numeric scalar")
    return float(array.item())


def _feature_fields(fields: dict, config: AudioConfig | None = None) -> dict:
    missing = set(_VECTOR_FIELDS).difference(fields)
    if missing:
        raise ValueError(f"missing feature fields: {', '.join(sorted(missing))}")
    if config is not None:
        grid = _common_grid(fields["frequency_hz"], config.frequency_hz)
    else:
        grid = _real_vector(fields["frequency_hz"], "frequency_hz")
        if grid.size < 2 or grid[0] < 0 or grid[-1] <= grid[0]:
            raise ValueError("frequency_hz must be an increasing uniform grid")
        grid = _common_grid(grid, np.linspace(grid[0], grid[-1], grid.size))
    result = {"frequency_hz": grid, "valid_mask": _mask_vector(fields["valid_mask"])}
    real_fields = ["magnitude_db", "transfer_real", "transfer_imag"]
    raw_fields = ("system_transfer_real", "system_transfer_imag")
    if any(fields.get(key) is not None for key in raw_fields):
        if not all(fields.get(key) is not None for key in raw_fields):
            raise ValueError("raw system response requires both real and imaginary parts")
        real_fields.extend(raw_fields)
    for key in real_fields:
        result[key] = _real_vector(fields[key], key)
    if any(a.shape != grid.shape for a in result.values()):
        raise ValueError("feature vectors must match the configured frequency grid shape")
    calibration = _text_scalar(fields.get("calibration", "none"), "calibration")
    if calibration not in _CALIBRATIONS:
        raise ValueError(f"unknown calibration mode {calibration!r}")
    result["calibration"] = calibration
    if fields.get("impulse_response") is not None:
        rir = _real_vector(fields["impulse_response"], "impulse_response")
        if config is not None and rir.size != config.rir_samples:
            raise ValueError("impulse_response length does not match the configured RIR window")
        result["impulse_response"] = rir
    if fields.get("alignment_method") is not None:
        method = _text_scalar(fields["alignment_method"], "alignment_method")
        if method not in {"marker", "correlation"}:
            raise ValueError(f"unknown alignment method {method!r}")
        result["alignment_method"] = method
    for key in ("timing_reference_seconds", "rir_start_seconds"):
        if fields.get(key) is not None:
            result[key] = _finite_scalar(fields[key], key)
    if fields.get("diagnostics") is not None:
        diagnostics = fields["diagnostics"]
        if not isinstance(diagnostics, dict):
            diagnostics = json.loads(_text_scalar(diagnostics, "diagnostics"))
        if not isinstance(diagnostics, dict):
            raise ValueError("diagnostics must be a JSON object")
        result["diagnostics"] = diagnostics
    return result


def save_feature(path: str | Path, feature: Feature) -> None:
    """Save a validated, pickle-free NPZ file, including available diagnostics."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("feature output must have the .npz extension")
    fields = _feature_fields(vars(feature))
    if "diagnostics" in fields:
        fields["diagnostics"] = json.dumps(fields["diagnostics"], allow_nan=False)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **fields)


def load_feature(path: str | Path, config: AudioConfig) -> Feature:
    """Load a shared-grid feature from NPZ or a SciPy-readable MATLAB v7 file."""
    path = Path(path)
    extension = path.suffix.lower()
    if extension == ".mat":
        fields = io.loadmat(path)
    elif extension == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            fields = dict(archive.items())
    else:
        raise ValueError("feature input must have the .npz or .mat extension")
    return Feature(**_feature_fields(fields, config))
