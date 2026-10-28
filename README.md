# CG-SVPNet: reproducible five-output forging prediction

Author-maintained reproduction project for contact-geometry-aware sub-voxel
material-point prediction. The independent application contains E0, E1 and E2
with **three displacement components, equivalent strain and equivalent stress**.
It preserves the original model parameters, processed data and case-wise split.

Authors: Dewang Chen and Jiangping Xu, School of Mechanical Engineering,
Jiangsu University. Correspondence: jiangpingxu@ujs.edu.cn.

## Reproduction scope

- E0: direct grid-to-particle interpolation of the five-channel grid output.
- E1: nonlinear material-point queries without local cell coordinates.
- E2: E1 with local cell coordinates and their squared/sinusoidal encodings.
- 21 closed-die cases; **15 training / 3 validation / 3 test cases**.
- Validation: cases 09, 06, 15 (75 target states).
- Test: cases 05, 16, 17 (74 target states).
- Fixed seed 20260805; the test set is not used for checkpoint selection.

This release does not contain every experiment in the paper: displacement-only
models, E2c/E5 variants, independent upsetting, and the original MPM simulation
platform are outside this E0/E1/E2 reproduction application. The published
processed data support surrogate training and inference; they do not replace
the solver configuration and source required to regenerate the MPM trajectories.

## Download data and original weights

Large binary files are distributed as assets of the pinned GitHub release
`v1.0.0`, **not through Git LFS and not inside Git history**. The complete
closed-die dataset contains 1,284 original files totaling 38,721,671,791 bytes.
The original file-level SHA-256 list is in `data/dataset_manifest.json`.
`release_manifest.json` lists the archive checksums and download locations.

After cloning this repository, use Python 3.13:

```console
python tools/download_release.py --scope weights
python tools/download_release.py --scope data
```

The downloader verifies each downloaded archive, extracts only safe relative
paths, and will not silently overwrite a different existing file. Downloads can
be repeated: already verified archives are reused. A failed partial archive is
retried from its beginning. Keep at least **90 GB free** for download archives,
the extracted data and working files; predictions need additional space.

Alternatively download the same assets manually from
<https://github.com/clawxiaobai/CG-SVPNet/releases/tag/v1.0.0>, place them in
`downloads/`, and run the downloader with `--offline`. Every data archive is
independently extractable into the repository root; all archives are needed for
the full split. No data are sampled, rescaled, recomputed or reassigned to a new
split during packaging.

## Install and launch

Windows 64-bit with Python 3.13 and Tcl/Tk is the tested desktop environment.
Run `setup_environment.cmd`, then `launch.cmd` or `启动程序.vbs`. Do not copy an
existing virtual environment from another computer. The installer uses the
official PyTorch 2.7.1 CUDA 11.8 wheels and the tested dependency lock.

The GUI selects the data directory, E0/E1/E2, training or existing weights,
validation/test export, field and time step. Training curves and validation
errors come from the current run. The full training profile retains the paper
settings; the reduced smoke profile is only an installation check.

For command-line use after environment installation:

```console
.venv\Scripts\python.exe -m forging.cli check --data-folder data/closed_die --output runs/data_check --full-hash
.venv\Scripts\python.exe tests/test_unit.py
.venv\Scripts\python.exe -m forging.cli predict --model E2 --data-folder data/closed_die --split test --format both --output runs/E2_test
.venv\Scripts\python.exe -m forging.cli train --model E2 --data-folder data/closed_die --split test --format both --output runs/E2_retrain
```

Choose a new output directory for each run. `--case case_16 --max-states 2`
restricts prediction to a small diagnostic subset, **not full-test evaluation**.
Training uses validation-only checkpoint selection and then exports predictions
for the chosen split. CPU operation is possible but full training is intended
for an NVIDIA GPU; the original device was an RTX 4060 with 8 GB memory.

## Reference results and actual verification

| Model | Displacement RMSE (mm) | Equivalent strain RMSE | Equivalent stress RMSE (MPa) |
| --- | ---: | ---: | ---: |
| E0 | 0.14425288 | 0.06927041 | 1.92186827 |
| E1 | 0.13717735 | 0.06161783 | 1.85552034 |
| E2 | 0.12388306 | 0.06157837 | 1.80479827 |

These are the archived full-test metrics for the original selected checkpoints
in `pretrained/<model>/final_metrics.json`, not fabricated outputs of a new run.
Displacement RMSE pools the three component errors, rather than taking the RMSE
of error-vector magnitudes. Equivalent strain is dimensionless; stress is MPa.
`Final` means progress >= 0.999 and may include multiple frames.

Prior application verification includes strict checkpoint loading, matching
initialization/forward/loss/gradients against the original implementation,
reduced one-epoch training for all three models, and full-particle inference on
the two final-window frames of case 16 with matching original metrics.
`qa/` records the completed checks. The independent project has **not** been
claimed to have rerun every 500-epoch training or every test state anew.
Retraining can differ across hardware and numerical environments.

Small preview figures are in `examples/`. Optional ParaView-ready final-frame
examples can be fetched with `python tools/download_release.py --scope examples`.
Predictions export NPZ, VTU and PVD with reference, prediction and error fields.
In ParaView open `sequence.pvd`; exported geometry uses reference positions.
To display predicted positions, apply Warp By Vector using
`prediction_minus_reference_position_mm` with scale factor 1, not the complete
displacement a second time.

## Method and data conventions

See `docs/METHOD.md`, `configs/dataset_split.json` and the Chinese
`README_老师复现.md`. The saved coordinate metadata is retained exactly. Its
working domain was derived from complete reference trajectories in the original
preprocessing; deployment on an unseen design using only initial geometry and
tool envelopes is a separate validation question. Target stress and strain are
supervision labels, not input features. The fixed descriptor coefficient 0.30
is an input-construction parameter, not a claim about physical solver friction.

The simulation data are numerical fields, not measurements from physical
experiments. Source provenance is retained in `reference/`. No additional
software or data licence has been selected by the authors in this release;
public availability must not be represented as a grant of an unspecified
third-party reuse licence.
