<div align="center">
<h1>Any4D: Unified Feed-Forward Metric <br>4D Reconstruction</h1>
<a href="assets/Any4D.pdf"><img src="https://img.shields.io/badge/Paper-blue" alt="Paper"></a>
<a href="https://arxiv.org/abs/2512.10935"><img src="https://img.shields.io/badge/arXiv-b31b1b" alt="arXiv"></a>
<a href="https://any-4d.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://github.com/Any-4D/Any4D"><img src="https://img.shields.io/badge/GitHub-Code-black" alt="Code"></a>
<a href="https://huggingface.co/spaces/theairlabcmu/Any4D"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<br>
<br>
<strong>
<a href="https://jaykarhade.github.io/">Jay Karhade</a>
&nbsp;&nbsp;
<a href="https://nik-v9.github.io/">Nikhil Keetha</a>
&nbsp;&nbsp;
<a href="https://infinity1096.github.io/">Yuchen Zhang</a>
&nbsp;&nbsp;
<a href="https://www.linkedin.com/in/tanisha-gupta-2a1934221/">Tanisha Gupta</a>
<br>
<a href="https://akashsharma02.github.io/">Akash Sharma</a>
&nbsp;&nbsp;
<a href="https://theairlab.org/team/sebastian/">Sebastian Scherer</a>
&nbsp;&nbsp;
<a href="https://www.cs.cmu.edu/~deva/">Deva Ramanan</a>
<br>
<br>
 Carnegie Mellon University
</strong>

</div>

<div align="center">

## Overview

**TLDR:** Any4D is a multi-view transformer for  
• Feed-forward • Dense • Metric-scale • Multi-modal  
4D reconstruction of dynamic scenes from RGB videos and diverse setups.

<img src="./assets/any4d_teaser_gif.gif" width="1000">

</div>


## Notes (12/12)

- The inference code will be refined and updated over the next few days.
- A stronger and more generalizable model checkpoint, along with full training code will be released soon.

Stay tuned for updates 


## Table of Contents

- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Models](#models)
  - [Sample Inference](#sample-inference)
- [Interactive Demos](#interactive-demos)
  - [Online Demo](#online-demo)
  - [Local Gradio Demo](#local-gradio-demo)
  - [Rerun Demo](#rerun-demo)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## Quick Start

### Installation
```bash
git clone https://github.com/Any-4D/Any4D.git
cd Any4D

# Create and activate conda environment
conda create -n any4d python=3.12 -y
conda activate any4d

# Optional: Install torch, torchvision & torchaudio specific to your system
# Install Any4D
pip install -e .

# For all optional dependencies
# See pyproject.toml for more details
pip install -e ".[all]"
pre-commit install
```

Note that we don't pin a specific version of PyTorch or CUDA in our requirements. Please feel free to install PyTorch based on your specific system.

#### Pixi (alternative)

A `pixi.toml` is provided as a reproducible alternative to conda. It pins PyTorch to the
CUDA 12.8 wheels, which are required for Blackwell GPUs (e.g. RTX 50 series):
```bash
pixi install
pixi run gpu-check          # sanity check: torch + CUDA device
pixi run download-checkpoint
pixi run python scripts/demo_inference.py --video_images_folder_path assets/stroller --viz
```

## Model Checkpoints

We release the pre-trained Any4D model checkpoint on Hugging Face and Google Drive:

**[🤗 HF Link](https://huggingface.co/airlabshare/any4d-checkpoint/resolve/main/any4d_4v_combined.pth)**

```bash
# Option 1: Hugging Face
mkdir -p checkpoints
wget -P checkpoints https://huggingface.co/airlabshare/any4d-checkpoint/resolve/main/any4d_4v_combined.pth

# Or use the helper script (also used by `pixi run download-checkpoint`)
python scripts/download_checkpoint.py --output-dir checkpoints
```

**[☁️ Google Drive Link](https://drive.google.com/drive/folders/1SOWr61vuv_bGtow6diAiWpIoUT50qSpk?usp=drive_link)**

```bash
# Option 2: Google Drive
mkdir -p checkpoints
cd checkpoints
gdown --folder https://drive.google.com/drive/folders/1SOWr61vuv_bGtow6diAiWpIoUT50qSpk
```


### Sample Inference

For quick example inference, you can run the following command:

```bash
# Terminal 1: Start the Rerun server
rerun serve --port 9877

# Terminal 2: Run Any4D demo
python scripts/demo_inference.py --video_images_folder_path assets/stroller --viz --port 9877
```

We provide multiple examples at [assets/example_images](assets/example_images). Please look at [Rerun Demo](#rerun-demo) for more control over visualization.

### RGB-D Inference (without MoGe)

If depth maps aligned to the RGB camera are available, `scripts/inference_rgbd.py` conditions the
model on them (task config `configs/model/task/rgbd.yaml`, which enables the depth and ray-direction
encoders) and uses the depth validity mask instead of the MoGe non-ambiguous mask. Depth maps are
expected to be single-channel images where `0` marks invalid pixels; use `--depth_scale` to convert
stored values to meters (`0.001` for millimeter `uint16` PNGs).

```bash
# Explicit directories
python scripts/inference_rgbd.py \
    --rgb_dir /path/to/rgb \
    --depth_dir /path/to/depth \
    --camera_params /path/to/rgb_camera_param.yaml \
    --checkpoint_path checkpoints/any4d_4v_combined.pth \
    --output_dir outputs/rgbd_run \
    --save_ply

# Or a session root that contains rgb/, mapped_depth/ and camera_parameters/
pixi run infer-rgbd --session /path/to/session \
    --start_idx 0 --end_idx 4 --output_dir outputs/rgbd_run
```

RGB and depth files are paired by file stem (a trailing `_rgb`/`_depth` suffix and a leading
`color_`/`depth_` prefix are ignored). Use `--start_idx`, `--end_idx`, `--stride` and `--chunk_size`
to select and batch the input views, `--task images_only` for RGB-only inference, or
`--no_amp` to disable bfloat16 autocast. Predictions (pointmaps, depth, poses, input depth, validity
masks) are written per view as compressed `.npz` files together with a `summary.json` that reports
the agreement between predicted and input depth.

### RGB-D Tracking Demo (Rerun)

`scripts/demo_rgbd_tracking.py` runs the same RGB-D inference and streams the reconstruction and
tracking results to [Rerun](https://rerun.io/). The fixed layout shows the current frame's RGB and
color-mapped input depth side by side above the metric point cloud and the trajectories of a fixed
set of reference-view points. As in `scripts/demo_inference.py`, the tracks are obtained by adding
the predicted world-frame scene flow of each view to the reference pointmap. Predicted depth and
all-frame overlays are opt-in via `--show_predicted_depth` and `--show_all_frames`.

Passing `--session <dir>` resolves `<dir>/rgb` (or `Color_*.jpg`), `<dir>/mapped_depth` and
`<dir>/camera_parameters/rgb_camera_param.yaml` automatically.

```bash
# Terminal 1: Start the Rerun server (or `pixi run rerun-serve`)
rerun serve --port 9877

# Terminal 2: Live view (connects to rerun+http://127.0.0.1:9877/proxy by default)
pixi run track-rgbd --session /path/to/session \
    --start_idx 0 --end_idx 240 --stride 30 --connect

# Save a recording and open it later (no viewer required while running)
pixi run track-rgbd --session /path/to/session \
    --start_idx 0 --end_idx 240 --stride 30 --save outputs/tracking.rrd
pixi run rerun-open outputs/tracking.rrd

# Explicit directories also work
pixi run track-rgbd \
    --rgb_dir /path/to/rgb --depth_dir /path/to/depth \
    --camera_params /path/to/rgb_camera_param.yaml \
    --save outputs/tracking.rrd
```


## Interactive Demos

We provide multiple interactive demos to try out Any4D!

### Online Demo

Try our online demo without installation: [🤗 Hugging Face Demo](https://huggingface.co/spaces/theairlabcmu/Any4D)

### Local Gradio Demo

We provide a script to launch our Gradio app. The interface and GUI allows you to upload image sequences/videos, run 4D reconstruction and interactively view them. You can launch this using:
```bash
# Install requirements for the app
pip install -e ".[gradio]"

# Launch app locally
python scripts/any4d_gradio.py
```

### Rerun Demo

We provide a demo script for interactive 4D visualization of metric reconstruction results using [Rerun](https://rerun.io/).
```bash
# Terminal 1: Start the Rerun server
rerun serve --port 9877 --web-viewer-port 9879

# Terminal 2: Run Any4D demo
python scripts/demo_inference.py \
    --image_folder /path/to/your/image/sequence \
    --checkpoint_path /path/to/your/checkpoint \
    --start_idx start_num \
    --end_idx end_num \
    --ref_img_idx ref_num \
    --ref_img_binary_mask_path /path/to/ref/image/binary/mask \
    --use_scene_flow_mask_refined True \
    --viz \
    --port 9877 \

# Terminal 3 or Local Machine: Open web viewer at http://127.0.0.1:9879 (You might need to port forward if using a remote server)
```

Optionally, if rerun is installed locally, local rerun viewer can be spawned using: `rerun --connect rerun+http://127.0.0.1:2004/proxy`.

## License
This repository is licensed under an open-source [Apache 2.0 license](LICENSE).

## Acknowledgments

We thank the following projects for their open-source code: [MapAnything](https://github.com/facebookresearch/map-anything), [DUSt3R](https://github.com/naver/dust3r), [MASt3R](https://github.com/naver/mast3r), [MoGe](https://github.com/microsoft/moge), [VGGT](https://github.com/facebookresearch/vggt), and [DINOv2](https://github.com/facebookresearch/dinov2).

## Citation

If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:
```bibtex
@article{karhade2025any4d,
  title={Any4D: Unified feed-forward metric 4d reconstruction},
  author={Karhade, Jay and Keetha, Nikhil and Zhang, Yuchen and Gupta, Tanisha and Sharma, Akash and Scherer, Sebastian and Ramanan, Deva},
  journal={arXiv preprint arXiv:2512.10935},
  year={2025}
}
```
