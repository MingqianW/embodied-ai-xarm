# DeltaAI MuJoCo Environment

## Supported baseline

- Python: 3.11 (the local validated interpreter is 3.11.15)
- Target: Linux aarch64
- MuJoCo: 3.2 or newer, below 4; local validation used 3.10.0
- Headless backend: EGL (`MUJOCO_GL=egl`)

Install the groups in `mujoco_deltaai_requirements.txt` into a new environment.
Do not copy or freeze the Windows Conda environment.

The dependency groups are:

- runtime: MuJoCo, NumPy, PyYAML, Pillow, headless OpenCV, imageio, and
  imageio-ffmpeg;
- conversion: LeRobot 0.1.0, datasets, huggingface-hub, PyArrow, and pandas;
- development/test: pytest;
- optional GUI: `opencv-python` and MuJoCo viewer dependencies. Do not install
  GUI OpenCV alongside `opencv-python-headless` on a headless node.

Install the OpenPI client from the pinned submodule rather than PyPI:

```bash
git submodule update --init third_party/openpi
python -m pip install -e third_party/openpi/packages/openpi-client
```

## Environment variables

```bash
export EMBODIED_AI_ROOT="$HOME/repos/embodied-ai-xarm"
export MUJOCO_OUTPUT_ROOT="/work/nvme/bfmk/mw89/mujoco_output"
export MUJOCO_DATASET_ROOT="/work/nvme/bfmk/mw89/mujoco_datasets"
export OPENPI_ROOT="$HOME/repos/openpi"
export OPENPI_CHECKPOINT_ROOT="/work/nvme/bfmk/mw89/openpi_checkpoints"
export HF_HOME="/work/nvme/bfmk/mw89/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export MUJOCO_GL=egl
```

Reusable code does not hard-code the DeltaAI username. The example values
above are deployment-specific and should be adjusted if the allocation or
account layout changes.

## Native EGL prerequisites

The exact package names depend on the DeltaAI node image and are not locally
verified. The node needs an NVIDIA driver compatible with its GPU, EGL/OpenGL
loader libraries, and a working headless EGL device. If MuJoCo reports
`GLAD`/EGL initialization errors, confirm the allocated node has a GPU and
that its driver libraries are visible before changing Python packages.

Run:

```bash
python scripts/check_deltaai_mujoco_environment.py --require-egl
python sim_mujoco/scripts/smoke_test_headless_render.py
```

The checker never contacts an OpenPI server.

## aarch64 cautions

- Verify that all selected versions publish Linux aarch64 wheels. MuJoCo,
  NumPy, Pillow, PyYAML, OpenCV, PyArrow, and LeRobot transitive dependencies
  are the most likely wheel/build constraints.
- `imageio-ffmpeg` may not bundle an aarch64 FFmpeg binary. Install a system
  `ffmpeg` module/package if needed; recorders fall back to PNG sequences when
  codecs are unavailable.
- LeRobot 0.1.0 is pinned to preserve the dataset schema currently used by
  this repository. Do not upgrade it during migration without rerunning
  converter and schema validation.
- OpenPI server dependencies are intentionally separate from this client and
  MuJoCo environment.
