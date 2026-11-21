# Acoustic Room Geometry Inference — Coding Agent Brief

## Objective and agreed scope

Build a university research prototype that estimates a room's enclosing geometry from a laptop-recorded sine sweep. Use Python, PyTorch, a 1D CNN, and PyRoomAcoustics. The CNN input is a processed frequency response. Its primary outputs are a fixed maximum number of candidate wall planes and their presence probabilities. Use supervised geometric labels and an optional simulator consistency loss with finite-difference gradients.

Use RealityScan only to obtain scaled reference geometry and measurement registration. Fit simple wall surfaces to those references and store their geometry numerically. Acquisition in Audacity and reference annotation may be manual. Implement the subsequent processing, training, evaluation, and inference as reproducible scripts.

The first version predicts the enclosing room shell, including recesses supported by its geometry representation. Omit furniture from output labels and simulator geometry. Real training and test recordings can contain furniture. Treat furniture, wall materials, and residual device response as sources of measurement variation. Describe results as room-shell inference from furnished-room recordings. The simulator loss is an approximate consistency constraint.

## Implementation choices

Put adjustable defaults in configuration and document them.

- Use log-magnitude frequency response as one CNN input channel, sampled on a uniform Hz grid. Preserve complex transfer functions and derived impulse responses as intermediate data.
- Use up to 12 vertical perimeter-wall candidates, with floor elevation and ceiling height stored separately. The 12-slot count excludes floor and ceiling. Support vertical walls and flat floors and ceilings.
- Preserve an ordered footprint in the labels so that the simulator can reconstruct wall extents and concave corners. Plane equations remain the primary reported wall representation. Add a small auxiliary output for boundary connectivity/order where required.
- Start with supervised training. Add the finite-difference physics term after the simulator and gradient checks pass. The final intended model uses the weighted combined loss.
- Use uncovered measurements for the baseline. Store cloth-covered takes as optional empirical references and evaluate the correction separately.

## Measurement files and metadata

For every measurement, retain the exact digital excitation WAV and untouched microphone WAV, including pre-sweep silence and the reverberation tail. Keep individual takes. Use uncompressed PCM or floating-point WAV and store actual sample rates. Store any reference-calibration WAV separately. Preserve original channels before selecting a microphone channel or resampling.

Record at least the following fields in a JSON or tabular manifest.

| Field group | Required information |
| --- | --- |
| Identity | Physical room ID, measurement ID, session ID, device/configuration ID, take ID |
| Audio | Excitation path, recording path, selected channels, actual sample rates, sweep start/end frequencies, sweep duration, recording duration |
| Device | Playback and recording device names, speaker routing, microphone selection, output volume, microphone gain, enhancement settings when available |
| Pose | Source and receiver coordinates, laptop orientation and lid configuration, coordinate-frame transform or registration reference |
| Calibration | Reference ID/path when present, reference method, settings, known reference propagation or its assumptions, correction version |
| Geometry | Geometry annotation path, metric scale, coordinate convention, footprint winding/order, floor and ceiling definitions |
| Conditions | Furniture arrangement ID or notes, door/window state, temperature if known, visible recording problems |

Record actual playback routing. A stereo laptop playing the same sweep from both speakers is a coherent two-source measurement. The simulator must represent that configuration, or data collection must consistently use an isolated speaker channel when the hardware supports it. For a multi-microphone laptop array, select and document a stable recording channel or store the channels separately. Keep device configuration fixed where practical.

Disable microphone automatic gain control, echo cancellation, noise suppression, spatial effects, and speaker equalization where accessible. Log settings that cannot be disabled. Use a repeatable moderate playback level and check clipping. Save several takes per pose when feasible. A repeatability check is more valuable than silently averaging inconsistent recordings.

## Existing MATLAB scripts

`sineSweep.m` intends to generate a 10-second linear sweep from 0 to 20 kHz at 44.1 kHz. To run it in MATLAB, replace `t**2` with `t.^2`, close the sine expression, use `fs` in `audiowrite`, and assign its declared return value or remove it. If Audacity creates the stimulus, use the exact exported WAV for deconvolution. Select the usable band from repeat recordings.

`dataProcessing.m` computes two FFTs and plots transformed magnitudes and their difference. It returns the string `done`. Its horizontal axis is FFT bin index, the sample rates are unused, and different recording lengths can produce incompatible grids. The outer absolute value in `abs(log10(abs(FFT)))` flips negative log values and changes the difference. Implement the processing below before using these results as features.

MATLAB can handle preprocessing. Export one SciPy-readable `.mat` file per measurement with MATLAB's `-v7` option. Include `frequency_hz`, `magnitude_db`, `valid_mask`, `transfer_real`, and `transfer_imag`. Record calibration mode and the feature definition in the manifest. Complex fields represent the measured system transfer unless complex calibration has been established. If the feature is a magnitude ratio alone, retain the uncorrected complex response separately. Python should validate the shared Hz grid and array shapes when loading these files. Use the same logical fields if processing WAVs in Python.

## Audio processing

Implement this pipeline in code. Use Audacity to acquire and inspect the audio, and use exported WAVs as processing inputs.

1. Validate file metadata, select channels explicitly, check clipping and silence, and estimate sweep position in the recording by correlation or a documented synchronization marker. Retain the full response tail.
2. Estimate the system transfer function from excitation and recording, using a regularized spectral division or a properly constructed exponential-sweep inverse filter. For spectral division, after defining common time axes and zero-padding sufficiently for linear convolution, use

   `H_system[k] = Y[k] * conj(X[k]) / (abs(X[k])**2 + epsilon[k])`.

   Choose the FFT length from excitation length and the supported impulse-response duration. Avoid circular wraparound. Make regularization relative to excitation energy or estimated noise and mask unexcited/unreliable frequencies. A finite sweep is a system-identification signal; the FFT magnitude of the microphone recording alone includes the excitation spectrum.
3. Preserve the complex response and an inverse-FFT impulse response. Record the timing correction, crop, taper, and effective response duration. Sweep alignment must preserve a documented acoustic timing reference if phase or arrival times are later used.
4. Apply a calibration only when its meaning is defined. In the simplified linear model, `Y = X * G_device * H_room`. A usable known device transfer function is removed through regularized complex division. If only magnitude calibration is available, subtract matched log-magnitude responses in dB. Subtracting recorded time waveforms or linear amplitudes does not generally invert this cascaded system.
5. Treat cloth-covered takes as an empirical reference. Covering the laptop can change the direct speaker-to-microphone path and introduce nearby reflections or frequency-dependent attenuation. Room sound can still reach the mic. Repeat covered and uncovered measurements at identical playback routing, gain, lid angle, and pose. For an optional feature comparison, subtract covered transfer magnitude in dB from uncovered transfer magnitude in dB on the same grid, masking bins with a weak reference. This yields a reference-relative magnitude. Use it in the simulator loss only if an equivalent reference normalization is justified. Compare it with the uncovered baseline on held-out rooms before adopting it.
6. Select the reliable frequency band after inspecting actual pilot recordings and repeats. Do not assume the laptop covers 20 Hz to 20 kHz accurately. Store `f_min`, `f_max`, bin spacing, noise masks, and frequency values with each processed dataset version. True frequency resolution depends on the usable response duration; zero-padding only interpolates it.
7. Produce `20 * log10(max(abs(H), magnitude_floor))` on a shared uniform Hz grid. Retain the detailed peaks and valleys. Any smoothing must be explicit and configurable. Avoid normalization that removes useful spectral structure. Optional removal of one scalar level offset per example can reduce gain variation, but must be applied identically in the simulated comparison and documented.
8. Fit feature-standardization statistics on the training split only. Store them with the checkpoint and use them unchanged at validation and inference. Keep input standardization separate from the physical spectral loss.

Raw recording here means the untouched microphone waveform before deconvolution, calibration, normalization, or denoising. Magnitude-only refers to a later feature selection. Keeping a waveform retains the information needed to estimate spectral phase relative to the saved excitation, subject to synchronization and device limitations.

## Geometry labels and representation

Store reference geometry in meters in a documented frame. Register the scan, source, receiver, and orientation consistently. A workable frame has z upward, an origin at a marked floor point or projected measurement reference, and x along a documented direction. Record the transform explicitly. If orientation is acoustically ambiguous, state the allowed symmetry/alignment in evaluation rather than mixing inconsistent labels.

Represent a wall plane as `n dot x + d = 0`, with unit normal `n` and signed distance parameter `d`. Normalize the plane consistently. Handle near-zero raw normal vectors explicitly. Store face winding and outward normals from the annotated boundary. Presence scales do not belong in the plane coefficients.

For each wall, store its finite face vertices or footprint endpoints, identity, and predecessor/successor in the enclosing boundary. Store an ordered simple footprint plus floor/ceiling values for vertical-room labels. Furniture is excluded from this boundary annotation. Record each interior pillar separately as a rectangular XY footprint with its vertical extent if available. A freestanding pillar has a separate boundary and stays outside the perimeter-wall slots. Trace any wall-attached protrusion in the footprint when it forms part of the surrounding boundary. Start with the outer shell in PyRoomAcoustics and flag pillar-present rooms in evaluation. If adding surveyed pillars later, test their occlusion and reflection effects and record whether their geometry is supplied at inference. Doors and windows are treated consistently with the measurement condition and the chosen room-shell simplification.

Plane equations alone leave finite wall extents and concave connectivity unspecified. The decoder must have an explicit source for that information. Do not reconstruct concave rooms by intersecting all wall half-spaces, which produces a convex region. Do not choose between ambiguous footprints using test-set reference geometry.

Keep the candidate planes and presence scores as the primary network output. For the initial vertical-room implementation, use an auxiliary boundary-order/connectivity head or another documented predicted boundary representation. A deterministic decoder uses selected candidates and predicted connectivity to construct finite faces. Validate the decoder on annotated rectangle and L-shaped examples before training. If the team accepts a restricted family of footprints, a small family classifier and family-specific decoder is another implementation option, but obtain agreement before restricting the dataset to those families.

Inference must return candidate planes, probabilities, selected planes, decoded finite geometry if valid, and an explicit validity status with reasons. Invalid decoded geometry remains a reported failure.

## CNN and supervised loss

Use a compact 1D CNN over frequency with several convolution blocks, nonlinearities, and moderate downsampling. Make its receptive field wide enough to observe multiple peaks. Preserve enough frequency-position information in the final feature representation to predict absolute scale. A modest flattened final feature map with a fully connected head is an acceptable starting point. Avoid extreme global pooling before verifying that relevant information survives.

Produce the following outputs.

- Plane parameters for `K_max` wall candidates.
- Presence logits for those candidates. Apply sigmoid to obtain reported probabilities.
- Floor/ceiling parameters under the vertical-room assumption.
- The agreed auxiliary boundary representation for obtaining finite faces.

Use a permutation-invariant assignment, such as Hungarian matching, between candidate and reference walls. The assignment cost should use normalized plane-angle and offset errors, with presence cost if useful. Compute geometric regression on matched existing walls. Treat unmatched candidates as absent with a balanced presence BCE loss. Do not regress arbitrary coordinates of absent target walls toward zero. Train any connectivity outputs using the same assignment so wall identities agree.

Use `L_model = w_plane * L_plane + w_presence * L_presence + w_boundary * L_boundary + w_height * L_height`, with weights and units documented. Enforce the chosen plane-sign convention or use sign-consistent plane comparisons. Presence probabilities need their own supervised loss because hard simulator selection supplies no useful ordinary gradient to those probabilities.

## Simulator consistency loss and gradients

Build PyRoomAcoustics rooms from valid finite faces or an ordered footprint extruded to the predicted height. Set source/receiver locations, routing, sample rate, sound speed, absorption assumptions, reflection order, and RIR duration explicitly. For the built-in laptop setup, record whether one or both speakers play and estimate their offsets from the selected microphone. Keep speaker and mic positions distinct. Record laptop height, heading, and rough offsets from two identifiable walls in the scan frame. Near-center placement alone gives an approximate pose. Test sensitivity to plausible pose changes before using a strong physics weight. Allow supervised use of examples whose pose is too uncertain for the physics comparison. Start with deterministic ISM, randomized image displacement disabled, and no added simulation noise. Pin the tested library version.

Transform the simulated RIR to the same spectral feature definition used for the measured target. For uncovered laptop recordings, residual device coloration limits direct comparison with bare simulated magnitude. Start with a repeatable band, optionally remove a smooth spectral envelope identically on both sides, and check sensitivity in a surveyed room before enabling the physics term on real recordings. Match the response window, frequency grid, calibration convention, valid-band mask, magnitude floor, and any scalar-level normalization. Do not compare a standardized CNN input to an unstandardized simulated magnitude.

Use the team's integrated absolute spectral difference in a defined domain, initially log magnitude in dB. For example,

`numerator = sum_k(w[k] * abs(A_sim[k] - A_measured[k]) * delta_f)`

`denominator = sum_k(w[k] * delta_f)`

`L_physics = numerator / denominator`

The mask/weights should downweight unreliable bands. Reject an example with an empty valid band instead of dividing by zero. Use fixed weights for a given measurement. Log loss units and avoid claiming that its numerical magnitude directly measures geometry error.

Use `L_total = L_model + lambda * L_physics`. Warm up with supervised loss and then ramp `lambda` from zero. Keep `lambda=0` as the required baseline. Use a documented fixed/global absorption value initially and log it. If fitting a small nuisance absorption parameter is needed later, constrain and regularize it rather than freely fitting a different spectrum for every wall.

For continuous geometry parameters `g_i`, estimate scalar-loss gradients by centered finite differences,

`dL/dg_i approximately equals (L(g + h_i*e_i) - L(g - h_i*e_i)) / (2*h_i)`.

Wrap the scalar simulator loss in a custom PyTorch autograd function, or inject its gradient with an equivalent documented vector-Jacobian operation. Differentiate continuous geometry outputs rather than CNN weights. Returning a detached loss tensor with `requires_grad=True` does not connect it to the CNN. Ensure the custom backward uses the incoming upstream scalar, including lambda and batch reduction factors.

Freeze selected wall membership, discrete connectivity, and matching during each local finite-difference evaluation. The estimate is conditional on that topology. Train discrete outputs with supervision. For an initial supervised physics refinement, using reference wall membership/connectivity during training is acceptable if explicitly reported as teacher forcing. Full inference must use predicted membership/connectivity, and its geometry validity and accuracy must be evaluated independently. Never use reference geometry to resolve inference topology.

Finite differences can perturb a plane out of verticality or break closure. Use only the independent continuous parameters supported by the valid decoder. If the decoder constrains normals or projects candidates, define the loss through that decoder and account for it consistently in forward and backward. Keep units clear for distances and angles. If validity changes across a perturbation, shrink the step within a bounded retry policy and log failures. Avoid silently substituting an unrelated room or inventing a zero acoustic residual.

Use supervised boundary/validity penalties and log the proportion of examples receiving physics loss. Invalid geometry may omit that example's simulator term while retaining supervised loss, but must remain counted in validation and in invalidity metrics so exclusion cannot improve the reported result. Select models primarily using geometry metrics and validity, not the loss on successful simulations alone.

Measure repeated evaluations at identical geometry and sweep several finite-difference step sizes on a tiny case. Check that small proposed continuous updates decrease the scalar loss in a smooth, fixed-topology case. Record simulator calls per training step and timing. Centered differences require roughly `2P` evaluations per example for `P` perturbed parameters, in addition to the forward evaluation. Make physics batch size and evaluation frequency configurable.

## Furniture handling

Use room-shell labels consistently for real recordings and synthetic data. Start with simulator data containing empty shells even when the measured room contains furniture. This introduces forward-model mismatch that should be reported and handled through a moderate physics weight and robust band weighting, rather than through extra predicted furniture boxes.

Do not assume furniture has negligible acoustic influence. Large reflecting or absorbing objects can alter reflections, scattering, and decay. As a small pilot check, record several repeated sweeps with the laptop fixed, move a table/chair or comparable movable object if practical, and repeat. Compare arrangement-induced feature changes with repeat-to-repeat changes using the chosen feature pipeline. Also check whether predicted wall geometry remains stable. If furniture movement produces material errors, retain the simplified output while documenting the reduced accuracy and collecting relevant variation where feasible.

## Training, evaluation, and acceptance

Split by physical room ID. Every pose, repeat, furniture arrangement, and session of one room belongs to a single split. Synthetic measurements from the same underlying room geometry must also stay together. With approximately 10 distinct rooms, use grouped cross-validation or leave-one-room-out where practical. Keep tuning inside the training and validation rooms for each evaluation. Report the number of rooms and per-room errors alongside aggregate metrics. Do not estimate held-out-room performance from repeat sweeps of training rooms.

Support synthetic pretraining with the same simulator/input processing, followed by supervised fine-tuning on measured examples. Randomize plausible materials and gain/device coloration only as documented synthetic augmentation. Keep the real/synthetic origin and calibration status of examples visible. Permit supervised real-data training directly when appropriate.

Evaluate plane angle and offset errors after assignment, wall presence precision/recall and count error, room height error, decoded geometry validity, and footprint IoU for valid vertical-room reconstructions. Report failures and all-room metrics alongside metrics conditional on valid decoding. Tune the presence threshold on validation data, then freeze it for testing.

Compare supervised training with physics-regularized training using the same splits and architecture. Save predicted geometry views and measured-versus-simulated spectral plots for representative successes and failures. Each result should identify its device configuration, calibration method, and held-out rooms.

Required regression checks should cover known-filter recovery from a synthetic sweep, calibration identity and band masking, permutation invariance of wall supervision, rectangle and concave L-shaped decoding, invalid-boundary rejection, a finite-difference gradient reaching CNN parameters, save/load consistency, and room-level split isolation. Test the actual finite-difference bridge on a tiny deterministic simulation, not only on a mock loss.

Provide documented commands for preprocessing a manifest, generating synthetic data, training, evaluating, and inferring from an excitation/recording pair. Inference should produce a machine-readable geometry result and a simple top-down/3D view. Save preprocessing configuration, feature statistics, split IDs, model settings, simulator settings, and random seeds with checkpoints. Keep the repository small and implementation readable.

## Implementation references

- Audacity Plot Spectrum describes amplitude-versus-frequency analysis and block averaging. Export original WAVs for the identification pipeline. https://manual.audacityteam.org/man/plot_spectrum.html
- Farina's measurement paper explains exponential sine sweep excitation and inverse filtering. https://www.angelofarina.it/Public/NordicSound-2007/NordicSound-Farina_paper.pdf
- PyRoomAcoustics documents finite polygonal faces, footprint extrusion, source/receiver configuration, and ISM settings. https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html

The project's uploaded RGI-Net paper is the architecture precedent for candidate wall planes and supervised presence probabilities. The boundary decoder and simulator-gradient bridge above are engineering requirements for this project's proposed integration, rather than capabilities established by that architecture alone.

- MathWorks documents MATLAB element-wise power and swept-frequency signal generation. https://www.mathworks.com/help/matlab/ref/double.power.html and https://www.mathworks.com/help/signal/ref/chirp.html
