# Implementation and input contracts

The paper's adopted path is **ACSH instance segmentation → directly predicted bounding box → local road-plane footprint → training-only calibration**. Decoded masks can be rendered and, with `--mask-area`, measured as a separate comparison. There is no tracking or temporal filtering in this implementation.

## Modules and retained mathematics

| Packaged module | Author implementation used |
| --- | --- |
| `models/acmix_modules.py` | `pothole_seg_refine/acmix_modules.py`; unchanged layers and forward equations |
| `models/base_architecture.py` | `pothole_seg_refine/base_architecture.py`; semantic transfer and parallel P2 construction |
| `models/refine_architecture.py` | `pothole_seg_refine/refine_architecture.py`; stabilized P2, P3 prototype input, identity-initialized ACmix |
| `geometry.py` | `runtime_acceleration/geometry_fast.py`, with `fit_plane` from `evaluate_local_plane_polygon_area_dry.py`, `mbtp_xy_area` from `evaluate_paper_mbpt_dry.py`, and `project_area` from `improved_end_to_end/infer.py` |
| `camera.py` | Parameterized camera maps and the 64-samples-per-edge bounding-box transform from `evaluate_bbox_calibrated_area_dry.py` |
| `calibration.py` | Route-specific log ordinary least squares from `improved_end_to_end/infer.py` |
| `training.py` | Segmentation architecture and optimization recipe from `pothole_seg_refine/experiment.py` |

The authors' numerical kernels retain the exterior ring enlarged by 1.5, even-coordinate decimation in global row-major order, seeded RANSAC (75 trials; distance threshold 0.20; at most 600 samples; minimum 100 input and 80 consensus points), SVD refit, plane projection, and dense X–Y triangle fallback. The bbox validity flag retains the original requirement for finite positive dense depth support even when the plane projection succeeds. This matters when an interior depth pixel is invalid.

The binary-mask comparison preserves contour hierarchy, subtracting holes, and uses the same supporting plane as the box. Its fallback integrates masked X–Y triangles. Neither fallback introduces the earlier ellipse factor: `mbtp_xy_area` contains that factor, which the bbox path cancels with `4/pi`, as in the frozen author pipeline.

Rectification maps are cached by camera model, all matrix/distortion bytes, rectified matrix, image size, and map representation. RGB uses `CV_16SC2` with linear interpolation to reproduce `cv2.fisheye.undistortImage` exactly; masks use `CV_32FC1` with nearest-neighbour interpolation, matching the original mask branch. The ROI ring produces exactly the original full-image OpenCV rectangle coordinates, including inclusive endpoints and clipping. Public input validation rejects invalid boxes, malformed cameras, and non-binary masks before computation.

## Camera and depth coordinates

`configs/m2s-camera.json` describes the released M2S-RoAD fisheye intrinsics/distortion at **1920 × 1208** pixels. Supply camera parameters for the actual input grid; the CLI rejects a mismatched resolution rather than silently scaling intrinsics. For other cameras, provide a separate JSON with `model` (`fisheye` or `pinhole`), `raw_k`, `distortion`, `image_size` (`[width,height]`), and optional `rect_k` and `balance`.

Depth is always a two-dimensional array on the **rectified** image grid. Use either an NPY containing depth, or an NPZ with `depth` and preferably `k` (the rectified 3 × 3 camera matrix). For live depth inference, `--depth-model` must point to a local Hugging Face metric-depth model directory; loading uses `local_files_only=True`. The paper used Depth Anything V2 Metric Outdoor Small. Resizing to the original rectified grid uses bicubic interpolation and `align_corners=False`.

With `geometry --camera ...`, input boxes and mask images are in the raw camera grid. With `geometry --rectified`, they are already rectified, and the NPZ must contain `k`. Optional `--image` follows the box coordinate system. Native inference takes raw RGB/BGR imagery and rectifies only the depth branch, as in the paper.

A boxes JSON can contain either `[[x0,y0,x1,y1], ...]` or `[{"box": [x0,y0,x1,y1], "confidence": 0.7, "mask": "optional-binary-mask.png"}, ...]`. Mask image paths are relative to that JSON. Nonzero mask pixels become binary foreground.

## Outputs and calibration

Per-frame JSON contains `box`, `confidence`, `geometry`, `bbox_m2`, and optional `mask_m2`. `geometry` retains the original `bbox_valid`, `bbox_route`, `bbox_raw`, `mask_valid`, `mask_route`, `mask_raw`, and `mask_pixels` fields. Inference also writes `predictions.jsonl` and a raw-image overlay for every retained frame. Source directories require unique image basenames. Cached multi-frame depth uses `<frame-name>.npz`; video names are `<video-stem>_<zero-based-frame-index:06d>`.

Raw geometry is an enclosing footprint in squared depth units. **It is not a calibrated pothole area.** If no coefficient file is supplied, `bbox_m2` remains null. The demo's 2.048 m² footprint is an analytic synthetic fixture with metre-valued depth, not a learned-model result.

Calibration uses `exp(intercept + slope * log(raw))` separately for `bbox_plane`, `bbox_fallback`, and optional `mask_plane` / `mask_fallback`. `calibrate` reads CSV, JSON, or JSONL with explicit `split`, `reference_m2`, and the geometry fields above (a nested `geometry` object is also accepted). It uses only `split=train`. A route needs at least two distinct positive raw areas; missing route coefficients remain unavailable instead of becoming an identity transform. Freeze the output before validation/test inference. Dataset-specific coefficients are not transferable guarantees for another camera, depth model, or domain.

## Training and checkpoint compatibility

`configs/acsh-segmentation.json` retains the selected `parallel_stable_p2_acmix_s1130` architecture and 60-epoch, 768-pixel configuration. The optional model dependencies pin Ultralytics 8.4.144 and Transformers 4.57.6 because the custom model uses Ultralytics internals. The inspected author environment used Python 3.11.16, PyTorch 2.7.1+cu126, NumPy 2.4.6, and OpenCV 5.0.0.

The checkpoint loader registers the legacy top-level names `acmix_modules`, `base_architecture`, and `refine_architecture` before loading trusted original author checkpoints. New checkpoints refer to the package's architecture classes. Weights and model directories must exist locally; the loader does not resolve a remote model name. Use the bundled loader for original checkpoints, rather than loading them through an unconfigured standalone Ultralytics process.

The training entrypoint retains the original optimizer, augmentation recipe, `amp=True`, and full-precision mask-AP model selection. Recipe overrides remain available. Ultralytics may request its comparison model for the upstream AMP compatibility check when training is run. The bundled checkpoint and historical paper numbers are not claimed to have been regenerated by this packaging pass. Training is not run by the demo or tests, and this repository does not automatically recreate the authors' private local dataset layout or the complete historical experiment registry.

## Verification

Run `python -m unittest discover -s tests -v` after an editable install. The eight CPU tests cover analytic footprints, robust plane fitting, invalid dense-support semantics, mask holes and fallback, rectification/resolution checks, exact ROI/full-grid ring equivalence, training-only route calibration, and pixel-exact cached RGB rectification.

The synthetic demo requires NumPy and OpenCV only. Neural model and live-depth inference require the optional `models` dependencies and local model inputs.

During release preparation on 2026-10-25, all eight tests passed. An editable install and its console entrypoint were checked in an isolated environment. The bundled checkpoint loaded on CPU; both cached-depth and live-depth inference processed one original-resolution test image and exported three valid predictions, per-frame JSON, JSONL and an overlay. Live depth used the existing frozen local snapshot. For three archived predicted boxes, box/mask geometry and calibrated area matched the original author implementation with maximum absolute difference **0.0**; RGB rectification was also pixel-identical. These are functional and equivalence checks, separate from the archived full-dataset results and runtime study.
