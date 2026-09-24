# Model weights

Pretrained model weights are not redistributed in this repository.

Validated file:

```text
vxm_dense_brain_T1_3D_mse.h5
SHA-256: 8e5fe6bcbca68b4fa867864460315fdaa7e00139cd522379cea79db5e63a9e3c
```

Obtain weights from the official VoxelMorph source and verify the checksum before inference.

The validated model is hosted by the FreeSurfer project:

```text
https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/models/vxm_dense_brain_T1_3D_mse.h5
```

Download it from a browser, or try the following command. The server may be
temporarily unavailable, so a command-line timeout is not evidence that the
file has been withdrawn.

```bash
curl -fL --retry 3 -o vxm_dense_brain_T1_3D_mse.h5 \
  https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/models/vxm_dense_brain_T1_3D_mse.h5
sha256sum vxm_dense_brain_T1_3D_mse.h5
```

The printed digest must match the value above. On PowerShell, use:

```powershell
Get-FileHash .\vxm_dense_brain_T1_3D_mse.h5 -Algorithm SHA256
```

The model, atlas, and test scan are upstream assets and are not distributed by
this repository. Their ownership and license terms remain with their respective
upstream providers; this project's MIT license does not relicense them.
