# Troubleshooting Notes

## Silent integer quantization after intensity scaling

### Symptom

An `mri_robust_register --mapmov` output may contain very few distinct integer values and little surviving foreground even though the transform appears plausible.

### Cause

If skull stripping preserves an integer datatype, intensity-scaled resampling can write small scaled values back to integer storage. The resulting rounding can severely quantize the output.

### Fix

Convert the skull-stripped source to `float32` before rigid or affine registration. The `preprocess_session.py` script performs this conversion automatically and verifies the saved datatype.

Use the audit command for any suspicious output:

```bash
python scripts/audit_nifti.py suspicious_output.nii.gz --fail-on-warning
```

Exit code `2` means the image failed the requested quality policy. For a controlled reproduction and comparison, use `scripts/run_dtype_ablation.py`; never use its experiment directory as production input.

## SynthStrip reports that CUDA is unavailable

The FreeSurfer-bundled Python environment may not have a CUDA-enabled PyTorch build even when TensorFlow detects the GPU. Run SynthStrip on the CPU by omitting `-g`. This does not prevent TensorFlow VoxelMorph inference from using the GPU.

## TensorFlow duplicate CUDA factory messages

Messages about registering cuDNN, cuFFT, or cuBLAS factories can appear during import. GPU availability must be verified with an actual TensorFlow operation rather than inferred from log severity alone.

## TensorRT warning

TensorRT is not required for this workflow. A missing TensorRT warning does not prevent ordinary TensorFlow GPU inference.

## NUMA warning under WSL

WSL kernels may not expose a NUMA node for the GPU. The warning is informational when TensorFlow lists the GPU and a test operation runs on `/GPU:0`.

## uv hard-link warning

When the uv cache is on the Linux filesystem and the virtual environment is on `/mnt/d`, hard links may be unavailable. Use `--link-mode=copy`. The installation is valid but consumes additional disk space.

## FreeSurfer registration convergence warning

Cross-subject registration to an atlas may reach the iteration limit before the default convergence threshold. Do not treat the warning alone as success or failure. Validate the saved transform and image using geometry checks, similarity metrics, affine plausibility, and visual inspection.

## Interpretation warning

Subtracting two subject-to-atlas displacement fields is useful for pipeline consistency QC but is not a validated longitudinal morphometry or atrophy measurement.
