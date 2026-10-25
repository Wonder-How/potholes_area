# Third-party software, models and data

The research code in this repository is distributed under **GNU AGPL version 3**; see [LICENSE](../LICENSE). The complete license text is copied unchanged from the installed Ultralytics distribution. The ACSH implementation extends Ultralytics classes and retains attribution to that project. The repository license does not replace upstream dependency licenses, grant additional dataset rights, or relicense separately obtained model weights.

## Software dependencies

The evidence below was read from the actual packages installed in the authors' experiment environment. Version numbers describe that archived environment, not a promise that every newer version is compatible. The `*.dist-info` entries name the license/metadata records inspected; environment-specific absolute paths are intentionally omitted.

| Component | Observed version | License evidence | Role |
|---|---|---|---|
| [Ultralytics](https://github.com/ultralytics/ultralytics) | 8.4.144 | `ultralytics-8.4.144.dist-info/METADATA`: `AGPL-3.0`; `licenses/LICENSE`: GNU AGPL v3 | YOLO training/inference, heads, losses and architecture base |
| [PyTorch](https://github.com/pytorch/pytorch) | 2.7.1+cu126 | `torch-2.7.1+cu126.dist-info/METADATA`: BSD-3-Clause; `LICENSE` and `NOTICE` retain component notices | Neural networks and tensor execution |
| [torchvision](https://github.com/pytorch/vision) | 0.22.1+cu126 | `torchvision-0.22.1+cu126.dist-info/LICENSE`: BSD-3-Clause | Vision operations used with the model runtime |
| [Transformers](https://github.com/huggingface/transformers) | 4.57.6 | `transformers-4.57.6.dist-info/licenses/LICENSE`: Apache-2.0 | Metric-depth model and processor loading |
| [NumPy](https://github.com/numpy/numpy) | 2.4.6 | `numpy-2.4.6.dist-info/licenses/LICENSE.txt`: BSD-3-Clause with bundled-component notices | Geometry and numerical operations |
| [OpenCV / opencv-python](https://github.com/opencv/opencv-python) | 5.0.0.93 | `opencv_python-5.0.0.93.dist-info/LICENSE.txt`: MIT packaging; `LICENSE-3RD-PARTY.txt`: Apache-2.0 for the included OpenCV library, plus component notices | Rectification, image operations and rendering |
| [Pillow](https://github.com/python-pillow/Pillow) | 12.3.0 | `pillow-12.3.0.dist-info/METADATA`: MIT-CMU; `licenses/LICENSE` includes additional bundled-component notices | Image/overlay and demonstration rendering |
| [PyYAML](https://github.com/yaml/pyyaml) | 6.0.3 | `pyyaml-6.0.3.dist-info/licenses/LICENSE`: MIT | Configuration support through Ultralytics |
| [SciPy](https://github.com/scipy/scipy) | 1.17.1 | `scipy-1.17.1.dist-info/LICENSE.txt`: BSD-3-Clause plus bundled-component notices | Original calibration/analysis utilities; not required for basic extracted geometry |
| [Matplotlib](https://github.com/matplotlib/matplotlib) | 3.11.1 | `matplotlib-3.11.1.dist-info/LICENSE`: Matplotlib license agreement and historical notices | Original plots/reference utilities |

Packages can include additional separately licensed libraries. Preserve their supplied license and notice files when redistributing those packages or binaries. This repository installs dependencies; it does not vendor their complete implementations or replace their notices with the table above.

FFmpeg was used as an external program to generate the demonstration GIFs. The media manifest records the observed FFmpeg build. No FFmpeg binary is bundled; use the terms and configuration of the FFmpeg build you obtain.

## Model and method attribution

- **YOLOv8 / Ultralytics**: Glenn Jocher, Ayush Chaurasia and Jing Qiu, *Ultralytics YOLOv8* (2023). The released ACSH checkpoint is an author-trained derivative of the Ultralytics model family. Its custom architecture definitions and fixed checksum are described in [PROVENANCE.md](PROVENANCE.md). The research-code AGPL notice must not be read as a new license grant over every upstream pretrained component or weight artifact.
- **ACmix**: Xuran Pan et al., *On the Integration of Self-Attention and Convolution*, CVPR 2022. The custom module implements the published convolution/attention combination and author-specific identity blending. Credit for the underlying ACmix method belongs to that work; this release does not claim the original method as a new invention. No separate upstream ACmix source-license file was present in the audited research module, so this document does not assign a license to an upstream ACmix repository.
- **Depth Anything V2**: Lihe Yang et al., *Depth Anything V2* (2024), arXiv:2406.09414. Inference uses `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`; its [publisher model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf) identifies the outdoor metric-depth variant. The [upstream project license statement](https://github.com/DepthAnything/Depth-Anything-V2#license), checked on 2026-10-25, identifies the Small model family as Apache-2.0 and distinguishes the Base/Large/Giant families, which use different terms. Obtain the selected weights from the publisher under the applicable upstream terms; the Transformers software license alone does not establish a model-weight license. Depth weights are not included here.
- **Prototype instance masks**: Daniel Bolya et al., *YOLACT: Real-Time Instance Segmentation*, ICCV 2019, provides background for prototype/coefficient mask representations. This attribution is to the method; YOLACT source code and weights are not bundled.

The included `weights/acsh-segmentation.pt` preserves the frozen research checkpoint. Code, checkpoint contents and upstream initialization are distinct artifacts. Consult the applicable upstream model terms for reuse; the AGPL text is not a blanket relicensing of third-party materials.

## Dataset and camera attribution

**M2S-RoAD**: Tzu-Yun Tseng, Hongyu Lyu, Josephine Li, Julie Stephany Berrio, Mao Shan and Stewart Worrall, *M2S-RoAD: Multi-Modal Semantic Segmentation for Road Damage Using Camera and LiDAR Data* (2025), arXiv:2504.10123. See the [original project](https://github.com/chinitaberrio/M2S-RoAD) for the release and its current terms. Camera constants in `configs/m2s-camera.json` trace to the project's published `toimage.py` example.

Download full images, polygons, LiDAR, odometry and other dataset assets from the original publisher under that publisher's terms. This repository does **not** upload or sublicense the dataset. Included GIFs are selected derived research visualizations with M2S-RoAD source attribution; their presence does not make the underlying dataset AGPL-licensed or confer dataset redistribution rights. The media manifest provides their sample and transformation provenance.

LiDAR-derived planimetric areas are research references constructed using released annotations/calibration and scan geometry. They are not surveyed ground truth or an official M2S-RoAD physical-area label release. The fixed 350/77/75 image split used in the manuscript was established by the authors of this work, not declared as an official dataset split.

Clear Road and Dark Road appear in the manuscript's additional detection experiments and originate from Boris Bučko et al., *Computer Vision Based Pothole Detection under Challenging Conditions*, Sensors 22(22):8878 (2022). Those datasets are not bundled, used as the metric-area calibration set, or included in this release's GIF sources. Obtain any such data under its original release terms.
