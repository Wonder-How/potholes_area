# Archived paper results

These numbers summarize the authors' current **unpublished manuscript**, *From Recognition to Measurement: Pothole Area Estimation through Local Road Geometry*. Preparing this repository did not retrain the models or rerun the full benchmark. The release packages the frozen implementation, checkpoint, calibration and selected visualizations; it is not a new independent replication.

## What is measured

The adopted path uses the ACSH instance segmenter's **directly predicted bounding box**, frozen monocular metric depth, fisheye rectification, local road-plane projection and training-only area calibration. Its decoded mask is a separate measurement comparison. The target is the pothole footprint on the local supporting-road plane, in square meters. It is not cavity surface area or volume.

LiDAR-derived planimetric references provide offline calibration and evaluation. They are independent of the monocular depth predictions, but remain subject to annotation, calibration, registration and plane-fitting uncertainty. Online inference takes RGB, fixed camera parameters and frozen calibration coefficients. The final evaluated pipeline has no tracking or temporal filtering.

## Protocols and denominators

| Protocol | Calibration / training | Evaluation | Meaning |
|---|---:|---:|---|
| Recognition image split | 350 train / 77 validation images | 75 test images | 895 / 154 / 146 pothole instances; ordinary and water-filled labels merged |
| Controlled geometry study | 257 area observations in 99 audited connected groups | 187 observations in 64 groups | Manual enclosing boxes; conditions on common usable geometry |
| Predicted-input area | 322 training area references | 58 reliable references in 75 test images | Ordinary, non-water-filled, quality grade A/B references |

The protocols reuse observations. In particular, 153 of the 187 geometry-study observations occur in recognition training images, so that cohort cannot measure held-out ACSH recognition or end-to-end performance. Audited connected groups stay together in the corresponding partitions. These splits have been explored previously; they are not an untouched external confirmation set.

For area evaluation, A20 and A50 are the percentages with absolute relative area error at most 20% and 50%. **End-to-end** A20/A50 keep all 58 references in the denominator, counting unmatched or invalid outputs as failures. **Matched** MAE and MdAPE use only valid matched outputs and therefore depend on each pipeline's matched subset. A20/A50 are distinct from detection AP.

## Controlled area geometry: manual boxes

All methods below share the 257 training / 187 held-out observation protocol. Calibrated methods use the same training-only log-OLS calibration family. This comparison evaluates the area module with manual localization, not automatic end-to-end performance.

| Estimator | MAE (m²) ↓ | MAPE (%) ↓ | MdAPE (%) ↓ | P90 APE (%) ↓ | A50 (%) ↑ |
|---|---:|---:|---:|---:|---:|
| Training-area median | 0.4407 | 82.85 | 51.84 | 185.62 | 49.73 |
| Original MBTP, raw image | 0.4109 | 75.96 | 65.30 | 85.94 | 30.48 |
| Rectified depth-XY surrogate | 0.6515 | 117.13 | 46.94 | 220.93 | 53.48 |
| MBTP + OLS | 0.3386 | 57.02 | 36.79 | 110.78 | 66.31 |
| Rectified depth-XY + OLS | 0.3661 | 58.92 | 41.24 | 114.68 | 62.57 |
| Local plane + Theil–Sen | 0.2119 | 36.68 | 24.40 | 67.84 | 82.89 |
| **Local plane + OLS** | **0.1983** | **32.40** | **23.33** | **55.39** | **86.10** |

The proposed estimator's MAE is 41.4% lower than calibrated MBTP and 45.8% lower than calibrated rectified depth-XY. Against the latter, the paired MAE difference is −0.1678 m², with pointwise 95% group-bootstrap interval [−0.3161, −0.0710] m². The bootstrap conditions on frozen upstream models and calibration; it does not account for all prior exploration.

## Recognition: three seeds

Test AP values below are percentages, mean ± sample SD over seeds 1130, 1131 and 1132. Detection uses a 40-epoch budget; segmentation uses 60 epochs. All variants use the same fixed image members and task-specific evaluation. The adopted area front end is the ACSH **segmentation** model.

| Configuration | Detector box AP50 | Detector box AP50–95 | Segmenter box AP50 | Segmenter box AP50–95 | Mask AP50 | Mask AP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n / YOLOv8n-seg | 23.15 ± 2.81 | 9.62 ± 1.28 | 27.58 ± 2.15 | 11.17 ± 0.57 | 29.29 ± 1.44 | 10.90 ± 0.50 |
| + ACmix | 24.18 ± 1.08 | 10.24 ± 0.46 | 24.67 ± 0.83 | 10.42 ± 0.88 | 25.40 ± 2.93 | 9.83 ± 0.79 |
| + P2 | 23.08 ± 1.84 | 10.24 ± 0.80 | 26.78 ± 1.08 | 11.80 ± 0.66 | 26.29 ± 1.51 | 10.16 ± 0.24 |
| ACSH, P2 + ACmix | 25.33 ± 2.00 | 11.51 ± 0.84 | 27.45 ± 3.02 | 12.19 ± 1.46 | 29.76 ± 3.54 | 11.08 ± 1.05 |

Mask-AP gains over the baseline are modest and variable across seeds. These observations do not establish a large segmentation gain or general cross-domain superiority. AP evaluation uses confidence 0.001, NMS IoU 0.70 and at most 300 predictions, which differs from the fixed area operating point.

## Predicted-input area: fixed seed 1130

This evaluation uses confidence **≥ 0.05**, box IoU **≥ 0.50**, confidence-descending one-to-one matching, NMS IoU 0.70 and at most 300 candidates. Every model uses its previously frozen seed-1130 checkpoint. These are not three-seed area averages.

| Model and geometry input | E2E A20 (%) ↑ | E2E A50 (%) ↑ | Matched MAE (m²) ↓ | Matched MdAPE (%) ↓ |
|---|---:|---:|---:|---:|
| YOLOv8n detector, box | 15.52 | 34.48 | 0.0914 | 28.23 |
| ACSH detector, box | 18.97 | **44.83** | 0.1118 | 28.66 |
| YOLOv8n-seg, box | 17.24 | 36.21 | 0.0844 | 28.08 |
| YOLOv8n-seg, mask | 17.24 | 32.76 | 0.0866 | 24.09 |
| ACSH segmenter, mask | **24.14** | 34.48 | 0.0945 | 30.38 |
| **ACSH segmenter, box: adopted** | **24.14** | 39.66 | **0.0832** | **21.53** |

The adopted path produces 29 valid matched areas from 58 references: 14 meet 20% error and 23 meet 50%. It improves strict A20 by 6.90 percentage points over YOLOv8n-seg plus box geometry, with paired 95% group-bootstrap interval [+1.75, +13.21] points. It shares the highest observed A20 and has the lowest conditional MAE among these pipelines; the ACSH detector has higher A50.

Conditional MAE comparisons require care. On 22 references valid in both segmentation box paths, MAE changes from 0.0837 to 0.0747 m², with paired difference interval [−0.02725, +0.00964] m² spanning zero. On the 15 references valid in all six pipelines, YOLOv8n-seg / ACSH box MAEs are 0.07312 / 0.07566 m². The matched-subset advantage is not uniform over all common cohorts.

## Runtime: final controlled single-frame comparison

The final manuscript uses the archived run identified as `m2s_single_frame_runtime_paired_no_kalman_20261023_controlled_thp_disabled`. Earlier runtime runs used different scopes or controls and should not be mixed with this table.

| Implementation | Mean ms/frame ↓ | Median ↓ | P90 ↓ | Mean-derived FPS ↑ |
|---|---:|---:|---:|---:|
| Original | 243.63 | 226.72 | 325.63 | 4.10 |
| Cached rectification + local support coordinates | **154.02** | **149.68** | **189.20** | **6.49** |

This is a **36.78%** mean latency reduction and **1.58×** speedup for the same frozen single-frame ACSH segmentation-to-box pipeline. The experiment uses 75 images × 3 passes = 225 paired executions per implementation, on a shared AMD EPYC 7K62 / RTX 4090 D server, PyTorch 2.7.1 / CUDA 12.6, FP32, batch 1, recognition size 768 and depth input 518 × 826. Both implementations share four-CPU affinity, four PyTorch/OpenCV/BLAS threads and passive worker waiting; transparent huge pages are disabled for the benchmark process only.

The timer covers stored-image read/decode, the full native-mask segmentation call, RGB rectification, **live depth inference for every execution**, predicted-box geometry, frozen calibration and final in-memory records. It excludes loading/warmup, one-time map initialization, camera acquisition before the stored image, result serialization/display and vehicle control. Native mask decoding remains in both recognition calls; auxiliary mask-area computation is excluded from both. No tracker or Kalman module is present, and no historical temporal-module time is subtracted.

All 225 pairs / 1,095 predicted-box records have identical numeric outputs and calibrated areas, with 1,032 valid area outputs per implementation. Rectified images and depth input tensors also match across all 75 images. Repeated passes reuse the same scenes; these timings are not independent hardware replications or a sensing-to-control latency guarantee.

## GIFs and display thresholds

The README demo comes from a previously rendered qualitative demonstration at confidence **0.25**, rather than the evaluation operating point of **0.05**. It retains the existing predictions, displayed numbers and reference comparisons. The source video presents independent annotated images with fixed holds; 24 FPS in that video and 10 FPS in the GIF describe playback only. It does not show continuous driving, tracking or measured inference throughput. See [the media manifest](assets/media_manifest.json) for source hashes, samples and transformations, including the additional archived exports.

The examples are curated visual explanations. Their errors cannot stand in for the full benchmark. Reference overlays labeled “GT” in the source presentations are manual annotations or LiDAR-derived references, not surveyed physical ground truth.

## Limits of the evidence

The available evidence concerns previously explored splits, usable geometry and non-water-filled reference targets. Calibration is scene/camera dependent; it is not an RGB zero-shot measurement guarantee. Local plane fitting, monocular depth, box localization and reference uncertainty interact. Independent data, water-filled targets, camera transfer, dense sequences and deployed latency remain future evaluation needs.
