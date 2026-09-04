# Methods

## Scope

The project studies a preprocessing failure mode: intensity scaling followed by
integer NIfTI storage can silently quantize a registered image. The failure was
initially observed while processing controlled-access imaging data. No raw or
derived participant-level data, measurements, paths, identifiers, or figures
from that work are distributed.

## Controlled dtype ablation

The local ablation holds image content, atlas, registration configuration, and
iteration limits constant while changing the moving-image storage dtype from
`int16` to `float32`. The integer branch intentionally reproduces the failure;
the floating-point branch represents the production-safe path. Generated
outputs remain local and are ignored by Git.

## Preprocessing and linear initialization

The optional full workflow supports DICOM-to-NIfTI conversion, skull stripping,
explicit `float32` conversion, rigid registration, and affine registration.
Production entry points reject integer-stored images before intensity-scaled
registration. Thresholds, interpolation choices, and registration commands are
implemented in the scripts and are unchanged by the public-data cleanup.

## Atlas construction

The official VoxelMorph atlas contains intensity and segmentation arrays with
shape `160 x 192 x 224` but no NIfTI affine. `scripts/prepare_atlas.py` derives
the documented crop geometry from a user-supplied FreeSurfer
`fsaverage/mri/orig.mgz` reference. This is a preprocessing convention, not
scanner metadata contained in the upstream archive.

## Single-pass deformable registration

The pinned VoxelMorph model is loaded through TensorFlow. In that implementation,
`VxmDense.references.y_source` is the moved image and
`VxmDense.references.pos_flow` is the final integrated, full-resolution
deformation used to generate it. The inference helper exposes both tensors as a
two-output Keras model, ensuring that the saved moved image and warp come from
one neural-network forward pass.

The deformation is stored in voxel units and VoxelMorph array-axis component
order. It is not a RAS-mm vector image.

## Public smoke demo

The public demo uses only the official VoxelMorph `atlas.npz`, `test_scan.npz`,
and separately acquired pretrained model. It measures full-array MSE and mean
Dice over the union of nonzero labels. The label map is transformed with
nearest-neighbor interpolation using the same final warp. Results are computed
at runtime and are not compared with hard-coded expected values.

## Quality control

The implementation provides:

- storage-dtype and intensity-range audits;
- checks for non-finite and constant images;
- checks for low sampled value diversity and extreme foreground sparsity;
- MSE and correlation on the union of robust foreground masks;
- intensity-derived foreground-mask Dice;
- displacement summaries;
- Jacobian determinant summaries and folding counts.

These quantities are engineering QC signals. They do not establish anatomical
ground truth, biological plausibility, clinical validity, or optimizer
convergence.

## Visual QC and controlled data

Visual inspection can be performed locally when permitted by the applicable
data agreement. Screenshots and participant-level visual assessments are not
publication artifacts in this repository.
