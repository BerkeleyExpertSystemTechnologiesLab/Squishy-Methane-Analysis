# Instrument profile v1

Status: **frozen specification**, 2026-09-13 UTC. Required companion to
[`beam_reconstruction_v1` / `mass90_column_v1`](reconstruction-policy-v1.md).
No numerical reference instrument has been selected.

The machine-readable structural schema is
[`instrument-profile-v1.schema.json`](instrument-profile-v1.schema.json).
The [worked-example profile](instrument-profile-v1.example.json) is explicitly
synthetic and exists only to make arithmetic reproducible. It is neither May's
hardware nor the current recovery demo's default. The production controller does
not yet load this profile format; see the [implementation handoff](../reviews/policy-v1-freeze.md).

## Required fields and units

All fields below are required; there are no implicit physical defaults. JSON
numbers must be finite. Unknown fields or unsupported versions are rejected.
The JSON Schema defines structure; the equations and compatibility rules below
are additional semantic validation requirements.

| Field | Meaning and constraint |
| --- | --- |
| `schema_version`, `artifact_type` | `1`, `instrument_profile`; not a controller transport message |
| `profile_id`, `provenance` | Stable identity, `synthetic` or `hardware_characterization`, and a nonempty source/assumption note |
| `coordinate_frame` | `normalized_fov_yx`, y down, x right, positions in `[0,1]^2` |
| `mapping` | Origin and positive span for both axes; declared `radian`, `degree`, or `synthetic_fov` units; affine map `q = origin + span * position` |
| `lattice_shape` | Positive row/column counts; cell centers `(row+0.5)/rows`, `(column+0.5)/columns`; pitches `1/rows`, `1/columns` FOV |
| `footprint` | Fixed rectangular height/width in FOV fractions, both in `(0,1]`; explicit clipped-FOV approximation |
| `scan_mode` | `column` or `point`; modes are separate named operating conditions |
| `mechanics.sweep_vertical_fov_per_second` | Fixed actual speed during an acquiring column, not an unspecified maximum; retained but unused for point mode |
| `mechanics.move_vertical_fov_per_second` | Fixed nonacquiring vertical transit speed |
| `mechanics.move_horizontal_fov_per_second` | Fixed horizontal transit speed; distance, including skipped columns, is charged |
| `mechanics.settling_seconds_per_visit` | Nonnegative additional delay once per acquiring column or point, after repositioning and before acquisition; zero explicitly denotes ideal mechanics |
| `acquisition.cadence_hz` | Commanded acquisition frequency during a pass; for point mode defines a fixed dwell slot |
| `acquisition.detector_capacity_hz` | Maximum sustainable readout frequency, independently characterized or synthetic |
| `acquisition.integration_seconds` | Positive exposure duration, at most one commanded cadence slot |
| `retrieval.quantity` | `beam_integrated_concentration`; controller receives the already retrieved scalar |
| `retrieval.units`, `retrieval.path_convention` | E.g. concentration-path units and whether round-trip correction is already applied; no automatic path-length division |
| `retrieval.calibration_id` | Frozen calibration provenance; numerical correction and residual-noise assumptions reside in the bound model configuration |
| `retrieval.temporal_model` | `exposure_midpoint_approximation`; acquisition interval retained despite this approximation |
| `retrieval.assumptions` | Explicit response/lag, footprint-edge, and residual error limitations; no claim of characterization from an ID alone |

The affine angular mapping is a v1 supported geometry. A nonlinear pointing map,
variable footprint, acceleration-limited motion, or sensor response spanning
several slots requires an explicit extended profile/model. A physical beam that
extends outside the modeled FOV does not actually clip itself. The clipped-FOV
likelihood is a declared approximation; a hardware profile must justify it or
remain nonconforming until a suitable edge model is implemented.

## Mechanical consistency

For column mode with row pitch `p_y=1/rows`, require

```text
cadence_hz = sweep_vertical_fov_per_second / p_y
cadence_hz <= detector_capacity_hz
0 < integration_seconds <= 1 / cadence_hz
```

Use a relative tolerance of `1e-10` for the cadence/speed equality. Merely being
below a detector's maximum does not define when the scanner measures. Do not
make pitch coarser, change the footprint, skip planned rows, or reduce actual
sweep speed to make an inconsistent profile work. Reject it with the failed
relation and required value. Integration response and motion still need separate
adequacy checks even when this arithmetic passes.

For `n` consecutive rows, the acquiring pass takes `n/cadence_hz` seconds.
The first reading midpoint occurs half a cadence slot after pass start. Use the
same speed in both vertical directions. Reposition horizontally then vertically;
charge each distance divided by its transit speed, plus one per-visit settling
delay. Both endpoints of a vertical span lie on lattice cell edges. Real
acceleration is deliberately absent from the ideal constant-speed v1 model and
must not be described as measured motor performance.

For point mode, travel to each selected cell center, settle, then occupy one
`1/cadence_hz` dwell slot and center the exposure in that slot. Do not fill
vertical gaps as extra observations in point mode. Use monotone columns and the
nearest vertical endpoint ordering specified for the compiler. The frozen
`mass90_column_v1` policy requires column mode; a point-mode comparator is a
separate policy/profile, not an interchangeable timing optimization.

Actual execution deviations use actual acquisition intervals, positions,
footprints, and receipts. An overrun is recorded; it does not retroactively make
the planned profile slower. The adapter must flag profile deviations and invalid
readout timing. Valid unexpected measurements may still be assimilated with an
explicit actual observation operator; they are not evidence that the run met its
planned mechanical contract.

## Run binding and immutable settings

Every experiment binds the full profile, formulation/policy identifiers, model
configuration, policy configuration, source/startup assumptions, initial pose,
session clock origin, causal wind mapping, and evaluation settings in one run
manifest. Record seeds and actual timing. Hash the profile's canonical JSON:
UTF-8 encoding, sorted keys, no whitespace separators, no NaN/Infinity, and no
implicit fields (`json.dumps(profile, sort_keys=True, separators=(",", ":"),
ensure_ascii=False, allow_nan=False)`). The digest is stored in the enclosing
manifest, not inside its own input. Do not promise cross-language semantic-number
canonicalization; external adapters must preserve these canonical bytes or use
the supplied hash. Raw file hashes may additionally identify artifacts.

All instrument settings are immutable within a run and shared across compared
policies. Model, policy, and evaluation configurations have separate hashes.
Different named profiles may be compared as different operating conditions, but
their results cannot establish an allocation-only advantage. Playback FPS, speed,
panel size, and display raster are observer settings outside the physical profile.

The model configuration must explicitly specify prior mean/covariance kernel,
inference grid, fixed calibration correction/noise, threshold, process noise,
diffusion, substep/footprint approximation, and source assumptions. The policy
configuration must supply the parameters enumerated in section 5 of the
formulation. A recorded JSON file containing only current Python defaults is
not automatically a complete instrument characterization.

## Historical mapping

| Evidence/configuration | What can be retained | What must not be inferred |
| --- | --- | --- |
| Recovered April 29-May 4 discussion | Narrow beam; vertical mechanical passes; horizontal movement between passes; beam pixel may not be enlarged | An independently specified detector cadence, integration response, or calibrated motor profile |
| Representative May code | 220 x 320 display; 1 s full-height pass and 0.05 s adjacent-column step as historical simulation settings | That every display row is a physically independent reading available at that speed |
| Earlier point-scan discussion | 20 Hz and 8-pixel footprint as a distinct exploratory point-scan condition | That these fill missing detector specifications in the later narrow-column model |
| July fast demonstration | Explicitly fast synthetic playback/operating condition | Reference hardware performance or a matched comparison with May |
| Recovery fixture | 12 x 16 inference, 20 x 32 physical lattice, 20 Hz limit, 0.025 s integration, 1 and 0.625 FOV/s speeds, 0.05 x 1/32 footprint | May's frozen beam, real calibration, or automatic conformance to the new commanded-cadence profile |

See the [historical scope review](../reviews/2026-09-12-scope-review.md) for
provenance. The raw recovered conversations and media remain local. No missing
hardware number is supplied by this freeze.
