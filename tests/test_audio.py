from dataclasses import replace

import numpy as np
import pytest
from scipy import signal
from scipy.io import savemat, wavfile

from rgi.audio import (
    AudioConfig, load_feature, process_wavs, save_feature, spectral_magnitude, spectrum_from_rir,
)


@pytest.fixture
def filter_wavs(tmp_path):
    config = AudioConfig(
        sample_rate=8000, f_min=300, f_max=2600, frequency_step_hz=25, rir_seconds=0.08,
        pre_arrival_seconds=0.01, taper_seconds=0.01,
    )
    n = 3200
    time = np.arange(n) / config.sample_rate
    sweep = 0.27 * signal.chirp(time, 140, time[-1], 3300) * signal.windows.tukey(n, 0.12)
    # Preserve file-time zero separately from the first active sweep sample.
    pcm = np.round(np.pad(sweep, (28, 0)) * 32768).astype(np.int16)
    taps = np.zeros(config.rir_samples)
    taps[[13, 33, 86, 430]] = [0.55, -0.18, 0.11, 0.075]
    marker = 336
    response = np.pad(signal.fftconvolve(pcm.astype(float) / 32768, taps), (marker, 0))
    excitation = tmp_path / "excitation.wav"
    recording = tmp_path / "recording.wav"
    wavfile.write(excitation, config.sample_rate, pcm)
    wavfile.write(recording, config.sample_rate, response.astype(np.float32))
    return config, excitation, recording, taps, marker


def test_recovers_known_filter_and_keeps_late_arrival(filter_wavs):
    config, exc, rec, taps, marker = filter_wavs
    feature = process_wavs(exc, rec, config)
    assert feature.alignment_method == "correlation"
    assert feature.timing_reference_seconds == pytest.approx((marker + 13) / config.sample_rate)
    shift = config.pre_arrival_samples - 13
    assert feature.rir_start_seconds == pytest.approx((marker - shift) / config.sample_rate)
    assert len(feature.impulse_response) == config.rir_samples
    assert np.argmax(np.abs(feature.impulse_response)) == config.pre_arrival_samples
    assert feature.impulse_response[430 + shift] > 0.035
    assert feature.valid_mask.all()
    np.testing.assert_array_equal(feature.frequency_hz, config.frequency_hz)
    indices = np.flatnonzero(taps)
    expected = np.exp(
        -2j * np.pi * np.outer(config.frequency_hz / config.sample_rate, indices + shift)
    ) @ taps[indices]
    np.testing.assert_allclose(feature.transfer, expected, atol=0.025, rtol=0.035)
    np.testing.assert_allclose(feature.magnitude_db, 20 * np.log10(np.abs(feature.transfer)))
    shifted_rir = np.pad(taps, (shift, 0))
    np.testing.assert_allclose(feature.transfer, spectrum_from_rir(shifted_rir, config), atol=0.025)


def test_marker_retains_delay_and_checks_linear_tail_coverage(filter_wavs, tmp_path):
    config, exc, rec, taps, marker = filter_wavs
    marker_seconds = marker / config.sample_rate
    feature = process_wavs(exc, rec, config, sweep_start_seconds=marker_seconds)
    assert feature.alignment_method == "marker"
    assert feature.timing_reference_seconds == pytest.approx(marker_seconds)
    assert feature.rir_start_seconds == pytest.approx(marker_seconds)
    assert feature.diagnostics["recording"]["pre_sweep_samples"] == marker
    assert feature.diagnostics["recording"]["pre_sweep_ac_rms"] == 0
    assert np.argmax(np.abs(feature.impulse_response)) == 13
    np.testing.assert_allclose(feature.transfer, spectrum_from_rir(taps, config), atol=0.025)
    rate, response = wavfile.read(rec)
    short = tmp_path / "short.wav"
    wavfile.write(short, rate, response[:-1])
    with pytest.raises(ValueError, match="response tail"):
        process_wavs(exc, short, config, sweep_start_seconds=marker_seconds)


def test_wav_validation_uses_selected_channel_and_each_files_rate(filter_wavs, tmp_path):
    config, exc, rec, _, marker = filter_wavs
    rate, y = wavfile.read(rec)
    upsampled = np.pad(signal.resample_poly(y, 2, 1), (0, 2))
    stereo = tmp_path / "stereo.wav"
    wavfile.write(stereo, 2 * rate, np.column_stack((np.ones_like(upsampled), upsampled)))
    processed = process_wavs(
        exc, stereo, config, excitation_rate=rate, recording_rate=2 * rate,
        recording_channel=1, sweep_start_seconds=marker / rate,
    )
    assert processed.diagnostics["excitation"]["sample_rate"] == rate
    assert processed.diagnostics["recording"]["sample_rate"] == 2 * rate
    assert processed.diagnostics["recording"]["channel"] == 1
    assert processed.diagnostics["recording"]["channels"] == 2
    baseline = process_wavs(exc, rec, config, sweep_start_seconds=marker / rate)
    np.testing.assert_allclose(processed.transfer, baseline.transfer, atol=0.02)
    # Upsampling cannot make frequencies above the original input Nyquist usable.
    extended = replace(config, sample_rate=2 * rate, f_max=4500, excitation_floor_db=-200)
    resampled = process_wavs(exc, stereo, extended, recording_channel=1, sweep_start_seconds=marker / rate)
    assert resampled.valid_mask[resampled.frequency_hz < rate / 2].any()
    assert not resampled.valid_mask[resampled.frequency_hz > rate / 2].any()
    with pytest.raises(ValueError, match="sample rate"):
        process_wavs(exc, stereo, config, recording_channel=1, recording_rate=rate)
    with pytest.raises(ValueError, match="sample rate"):
        process_wavs(exc, rec, config, excitation_rate=2 * rate)
    with pytest.raises(ValueError, match="channel .*unavailable"):
        process_wavs(exc, rec, config, excitation_channel=1)
    with pytest.raises(ValueError, match="clipped"):
        process_wavs(exc, stereo, config, recording_channel=0)
    silent = tmp_path / "silent.wav"
    wavfile.write(silent, rate, np.full(y.shape, 128, dtype=np.uint8))
    with pytest.raises(ValueError, match="silent or constant"):
        process_wavs(exc, silent, config)
    positive_rail = tmp_path / "positive_rail.wav"
    u8 = np.full(y.shape, 128, dtype=np.uint8)
    u8[0] = 255
    wavfile.write(positive_rail, rate, u8)
    with pytest.raises(ValueError, match="clipped"):
        process_wavs(exc, positive_rail, config)


def test_excitation_power_masks_unexcited_bins(filter_wavs):
    config, exc, rec, _, _ = filter_wavs
    config = replace(config, f_min=0, f_max=4000, excitation_floor_db=-25)
    processed = process_wavs(exc, rec, config)
    bins = config.frequency_hz
    assert processed.valid_mask[(bins >= 400) & (bins <= 2600)].all()
    assert not processed.valid_mask[bins >= 3700].any()
    assert np.isfinite(processed.magnitude_db).all()


def test_complex_identity_preserves_raw_and_masks_weak_reference(filter_wavs):
    config, exc, rec, _, _ = filter_wavs
    baseline = process_wavs(exc, rec, config)
    device = baseline.transfer
    device[2] *= 1e-5
    valid = baseline.valid_mask.copy()
    valid[5] = False
    reference = replace(baseline, transfer_real=device.real, transfer_imag=device.imag, valid_mask=valid)
    corrected = process_wavs(exc, rec, config, calibration="complex", reference=reference)
    keep = baseline.valid_mask.copy()
    keep[[2, 5]] = False
    np.testing.assert_array_equal(corrected.valid_mask, keep)
    np.testing.assert_allclose(corrected.transfer[keep], 1, atol=2e-6)
    np.testing.assert_allclose(corrected.magnitude_db[keep], 0, atol=2e-5)
    np.testing.assert_array_equal(corrected.system_transfer_real, baseline.transfer_real)
    np.testing.assert_array_equal(corrected.system_transfer_imag, baseline.transfer_imag)
    np.testing.assert_array_equal(corrected.impulse_response, baseline.impulse_response)
    dark = replace(reference, transfer_real=np.zeros_like(device.real), transfer_imag=np.zeros_like(device.imag))
    with pytest.raises(ValueError, match="empty valid frequency band"):
        process_wavs(exc, rec, config, calibration="complex", reference=dark)


@pytest.mark.parametrize("mode", ["magnitude", "covered"])
def test_db_calibration_is_reference_relative_and_keeps_system_phase(filter_wavs, mode):
    config, exc, rec, _, _ = filter_wavs
    baseline = process_wavs(exc, rec, config)
    device = -0.5j * baseline.transfer
    device[4] = 0
    reference = replace(
        baseline, transfer_real=device.real, transfer_imag=device.imag,
        magnitude_db=np.full_like(baseline.magnitude_db, 123),
    )
    corrected = process_wavs(exc, rec, config, calibration=mode, reference=reference)
    assert corrected.calibration == mode
    assert not corrected.valid_mask[4]
    assert np.count_nonzero(~corrected.valid_mask) == 1
    np.testing.assert_allclose(corrected.magnitude_db[corrected.valid_mask], 20 * np.log10(2))
    np.testing.assert_array_equal(corrected.transfer, baseline.transfer)


def test_shared_spectrum_windows_rir_and_applies_masked_transforms(filter_wavs):
    config, exc, rec, _, _ = filter_wavs
    # A response outside the supported duration and one exactly at the last
    # (zero-weight) taper sample must not enter the comparison spectrum.
    rir = np.zeros(config.rir_samples + 4)
    rir[0] = 0.5
    rir[config.rir_samples - 1:] = 1
    np.testing.assert_allclose(spectrum_from_rir(rir, config), 0.5)
    np.testing.assert_allclose(spectrum_from_rir(rir[:1], config), 0.5)
    config = replace(config, remove_scalar_level=True, envelope_smoothing_hz=140)
    feature = process_wavs(exc, rec, config)
    np.testing.assert_allclose(feature.magnitude_db, spectral_magnitude(feature.transfer, feature.valid_mask, config))
    mask = feature.valid_mask.copy()
    mask[11] = False
    perturbed = feature.transfer
    perturbed[11] = 1e12
    target = spectral_magnitude(feature.transfer, mask, config)
    np.testing.assert_allclose(spectral_magnitude(perturbed, mask, config)[mask], target[mask])
    assert target[mask].mean() == pytest.approx(0, abs=1e-13)


def test_npz_and_mat_round_trip_and_validation(filter_wavs, tmp_path):
    config, exc, rec, _, _ = filter_wavs
    baseline = process_wavs(exc, rec, config)
    feature = process_wavs(exc, rec, config, calibration="complex", reference=baseline)
    npz = tmp_path / "feature.npz"
    save_feature(npz, feature)
    restored = load_feature(npz, config)
    for key, value in vars(feature).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(restored, key), value)
        else:
            assert getattr(restored, key) == value

    mat_data = {
        "frequency_hz": feature.frequency_hz[:, None],
        "magnitude_db": baseline.magnitude_db[None, :],
        "valid_mask": baseline.valid_mask[:, None].astype(np.uint8),
        "transfer_real": baseline.transfer_real,
        "transfer_imag": baseline.transfer_imag,
    }
    mat = tmp_path / "feature.mat"
    savemat(mat, mat_data)
    imported = load_feature(mat, config)
    assert imported.calibration == "none"
    np.testing.assert_array_equal(imported.valid_mask, baseline.valid_mask)
    np.testing.assert_array_equal(imported.transfer, baseline.transfer)
    savemat(mat, {**mat_data, "calibration": "magnitude"})
    assert load_feature(mat, config).calibration == "magnitude"
    savemat(mat, {**mat_data, "frequency_hz": feature.frequency_hz + 1})
    with pytest.raises(ValueError, match="uniform frequency grid"):
        load_feature(mat, config)
    savemat(mat, {**mat_data, "magnitude_db": baseline.magnitude_db.reshape(3, -1)})
    with pytest.raises(ValueError, match="one-dimensional"):
        load_feature(mat, config)
    savemat(mat, {**mat_data, "valid_mask": np.full_like(baseline.magnitude_db, 0.5)})
    with pytest.raises(ValueError, match="0/1"):
        load_feature(mat, config)
