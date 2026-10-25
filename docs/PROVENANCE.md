# Implementation and artifact provenance

This release reorganizes the authors' existing research implementation into a portable package. The current manuscript is *From Recognition to Measurement: Pothole Area Estimation through Local Road Geometry* by Dehao Wang, Haohang Zhu, Bowen Xu, Yiwen Xu and Kaiqi Liu. It is unpublished; this repository does not assign a DOI, journal or acceptance status.

The older draft emphasized MBTP, tracking and Kalman smoothing. The released principal path follows the current single-frame **ACSH segmentation → predicted box → local road geometry → calibrated area** method. Old temporal-method results are not advertised as current-pipeline results.

## Source map

The “research source” column uses relative archive names for traceability. It does not expose server locations or imply that all experiment scripts are included in this release.

| Released module / artifact | Research source | Packaging change |
|---|---|---|
| `src/pothole_measurement/models/acmix_modules.py` | `pothole_seg_refine/acmix_modules.py` | Preserve custom ACmix classes; adapt package imports |
| `src/pothole_measurement/models/base_architecture.py` | `pothole_seg_refine/base_architecture.py` | Preserve P2 graph and semantic pretrained-weight transfer; adapt imports |
| `src/pothole_measurement/models/refine_architecture.py` | `pothole_seg_refine/refine_architecture.py` | Preserve parallel P2, prototype branch and custom segmentation classes; adapt imports |
| `src/pothole_measurement/models/__init__.py` | New release wrapper | Lazy loading and legacy module aliases for archived checkpoint deserialization |
| `src/pothole_measurement/geometry.py` | `p1130_area_validation/evaluate_local_plane_polygon_area_dry.py`; `evaluate_paper_mbpt_dry.py`; `improved_end_to_end/infer.py`; `runtime_acceleration/geometry_fast.py` | Extract plane fit, legacy XY surrogate, projection, and equivalent local-ring implementation; parameterize camera input |
| `src/pothole_measurement/camera.py` | Box-boundary transform from `p1130_area_validation/evaluate_bbox_calibrated_area_dry.py`; camera-map approach from `runtime_acceleration/rectify_cache.py` | Portable camera wrapper and cached rectification |
| `src/pothole_measurement/calibration.py` | `p1130_area_validation/improved_end_to_end/infer.py`, `calibrate` / `area` | Extract route-specific log-OLS fit and calibrated output |
| `src/pothole_measurement/inference.py` | `p1130_area_validation/improved_end_to_end/infer.py`, `cache` / `predict` | New local-path loading, inference, overlays and JSON output wrapper |
| `src/pothole_measurement/training.py` | `pothole_seg_refine/experiment.py`, `RefineTrainer` / training arguments | Portable training wrapper; no claim that packaging retrains or replicates all ablations |
| `src/pothole_measurement/cli.py` | New release wrapper | Expose geometry, inference, calibration, training and demo commands |
| `configs/acsh-segmentation.json` | Frozen `parallel_stable_p2_acmix_s1130` run configuration | Carry selected architecture and training settings |
| `configs/m2s-camera.json` | `p1130_area_validation/m2s_lidar_reference_area.py`; originally the M2S-RoAD `toimage.py` calibration example | Separate fixed camera constants from the implementation |
| `configs/m2s-calibration.json` | `m2s_improved_end_to_end/calibration.json` | Preserve frozen calibration coefficients |
| `weights/acsh-segmentation.pt` | `p2_acmix_segrefine_20261017/parallel_stable_p2_acmix_s1130/weights/best.pt` | Rename for distribution; do not modify checkpoint contents |
| `scripts/make_demo_gifs.py`, `docs/assets/*.gif` | Previously rendered area and recognition demonstration videos | Extract temporal intervals, sample frames, resize proportionally and quantize GIF palettes; remove the header title in the README GIF and its static preview |
| `docs/assets/paper/structure_revised.png`, `structure_revised.pdf` | Current manuscript `Figure/structure_revised.png` and `Figure/structure_revised.pdf` | Exact file copies of the original framework figure; no redrawing or restyling |

The archived checkpoint contains Python class references to the original module names. Compatibility aliases retain those names while the release code lives in `pothole_measurement.models`. Changing class definitions or silently substituting a different architecture can make a checkpoint load incorrectly even when a filename is unchanged.

## Frozen model

| Field | Value |
|---|---|
| Architecture | YOLOv8n-seg with supervised parallel P2 branch, stabilized P2 feature blending and identity-initialized ACmix on the prototype input |
| Seed | 1130 |
| Training budget | 60 epochs, image size 768 |
| Final options | `stable_p2=True`, `detail=False`, `output_detail=False`, `dice_weight=0` |
| Released checkpoint | `weights/acsh-segmentation.pt` |
| SHA-256 | `e5a809756eb4cdc0b86ee45343925a40cac2dcf7ead5311d23219ec5d7735f2a` |
| Frozen depth model | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` |

The segmenter checkpoint was frozen before the reported area comparison; it was not selected by the area-test score. The depth checkpoint is obtained separately from its upstream publisher, rather than bundled in this repository. The archived depth-cache snapshot identifier was `fd2c22027eaf20374204f14099b8341e1925ad39`.

## Calibration and geometric behavior

The distributed calibration belongs to the fixed-image predicted-input experiment: **322 training area references / 58 test references**. It is distinct from the controlled **257 / 187** manual-box geometry study. Mask calibration has 321 usable training geometries: 286 plane and 35 fallback. It uses manual training-mask geometry, not test-mask correction.

For route `r`, the output in m² is `exp(a_r + b_r * log(raw_area_m2))`. Frozen coefficients are:

| Route | Intercept `a_r` | Slope `b_r` |
|---|---:|---:|
| `bbox_plane` | −2.32704758435423 | 0.8060188352912152 |
| `bbox_fallback` | 0.6718630506599197 | 0.9104895859780955 |
| `mask_plane` | −1.8763892717697115 | 0.9522206277265426 |
| `mask_fallback` | 1.042186111753218 | 0.9111272379162225 |

The source calibration file's SHA-256 is `a6a4a1665b7d5cf37bac11de9d131e2b23b1856fcdede517f443441ef3b935e2`. Numerical coefficients are the essential calibration identity; formatting changes can change a JSON file hash.

The estimator samples 64 points per raw box edge for rectification, fits the surrounding 1.5× box ring at even pixel coordinates, and retains the original finite-positive dense-depth validity gate. Plane fitting requires at least 100 support points, subsamples to at most 600, runs 75 hypotheses with 0.20 m inlier threshold, requires 80 inliers and refits with SVD. Random sampling order and seed affect the fitted plane.

If plane fitting or intersection fails, the fallback retains the legacy depth-XY accumulation with its ellipse multiplier removed. That accumulation is an empirical surrogate, not an exact partition or a cavity surface reconstruction. The dense validity gate also applies to a successful plane route; a fallback does not guarantee an area for every box. These behaviors are retained to keep archived coefficients and results interpretable.

Fixed camera calibration and learned area calibration are separate requirements. Replacing the camera, metric-depth model, box geometry, support sampling or fallback formula requires fresh validation and may require new training-only calibration. Do not reuse these coefficients as universal scene-independent constants.

## Media and result records

[RESULTS.md](RESULTS.md) records the manuscript's metrics, splits, operating thresholds and timing scope. It explicitly distinguishes archived results from tests performed during repository preparation.

[The media manifest](assets/media_manifest.json) records the original video filenames and hashes, checkpoint hash, selected sample IDs, confidence 0.25 display threshold, extraction intervals, dimensions and output hashes. The GIFs retain source predictions and numbers. They are curated examples based on independent annotated images, not continuous video or inference-speed evidence. The area benchmark instead operates at confidence 0.05.

The archived runtime result used for the current manuscript is identified as `m2s_single_frame_runtime_paired_no_kalman_20261023_controlled_thp_disabled`. It measures the same single-frame box-area pipeline before and after equivalent implementation changes; auxiliary mask area, tracking and Kalman filtering are outside both timed implementations.

Only code, selected demonstration media, frozen model/configuration artifacts and concise result summaries are intended for this release. Dataset archives, full manuscript/PDF, internal review correspondence and unrelated experiment outputs are excluded.
