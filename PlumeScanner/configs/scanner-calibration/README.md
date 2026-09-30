# Frozen synthetic calibration candidate

These two complete run configurations reproduce the seed-101 continuous case for
the coarse and doubled physical profiles. The selected model uses a 6 s half-life,
source width .035 FOV and rate .2829738193616142 synthetic units/s. Margin remains
zero after unsuccessful padding development. Other model/instrument settings stay
fixed. Decode with `full_fov_execution.decode_config`; pass only `config.session`
to `SensingController.from_session` for the public acquisition API.

They are experimental references, not a promoted baseline. The candidate improves
tested reconstruction losses but falsely sustains focus in blank scenes and after
cessation. Real sensor calibration and uncertainty calibration remain open.
See the [review](../../docs/reviews/scanner-baseline-calibration.md) and
[protocol](../../docs/specs/scanner-baseline-calibration.md). The complete 72-case
bound manifest and execution artifacts remain local under
`outputs/scanner-calibration/`; robotics integration is deferred.
