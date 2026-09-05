# Public VoxelMorph validation

This validation runs the VoxelMorph atlas-registration pipeline on four T1-weighted scans from the public CC0 OpenNeuro dataset `ds005125` (version `v1.0.0`, DOI `10.18112/openneuro.ds005125.v1.0.0`). It is a software reproducibility check, not clinical validation or population-level generalization.

Inputs were reoriented from their affines, linearly resampled to the atlas grid, converted to `float32`, and robustly nonzero min-max scaled to `[0, 1]`. The four independent registrations were atlas ← sub-01 through atlas ← sub-04.

The intensity-derived Dice is a foreground quality-control metric, not anatomical ground-truth Dice. Jacobian non-positive percentages are reported per scan and do not independently establish registration correctness. No biological or clinical conclusion is claimed.

Raw MRI, model weights, atlas, warp fields, and full-resolution outputs are not distributed in GitHub. The earlier ADNI experiment remains a separate, limited, unchanged case study. This public validation complements, and does not replace, that experiment.

See `public_validation_metrics.json` for sanitized provenance and measured results.
