# Failure Analysis: Silent Integer Quantization

## Incident class

An intensity-scaled registration can complete successfully, write a transform,
and produce a readable image while destroying useful intensity information.
The defect is silent because file validity and process exit status do not test
whether the stored numerical representation preserved the resampled values.

The issue was initially observed during controlled-access imaging work. No
participant-level evidence or derived measurements from that work are included
in the public repository.

## Mechanism

When scaled floating-point values are written using integer NIfTI storage,
rounding can collapse the output into very few discrete values. The spatial
transform may still exist, so the failure must be detected through dtype and
intensity-distribution checks rather than process completion alone.

## Controlled reproduction

`scripts/run_dtype_ablation.py` preserves a local controlled comparison:

| Branch | Stored dtype | Registration stage | Purpose |
|---|---|---|---|
| A1 | int16 | rigid | Reproduce quantization risk |
| A2 | int16 | affine | Observe propagated damage |
| B | float32 | rigid | Safe initialization path |
| C | float32 | affine | Safe linear alignment path |
| D | float32 | affine + VoxelMorph | Deformable stage |

Outputs from this experiment are ignored and must remain private when they are
derived from controlled-access data. Public automated evidence comes from
synthetic regression tests; public integration evidence comes from official
VoxelMorph example assets.

## Prevention

1. Convert the skull-stripped moving image to `float32` before registration.
2. Reject integer-stored production inputs.
3. Audit finite values, dynamic range, sampled value diversity, and foreground
   fraction after resampling.
4. Treat similarity, displacement, Jacobian, and visual checks as complementary
   QC signals rather than proof of correctness.

## Interpretation boundary

Preventing quantization demonstrates preservation of numerical signal. It does
not establish anatomical accuracy, clinical validity, or biological change.
