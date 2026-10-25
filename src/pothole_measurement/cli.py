"""Command line workflows. Geometry and demo never load neural networks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .camera import Camera
from .calibration import fit_calibration, read_records
from .inference import load_depth, load_boxes, measure_records, read_image, write_frame


def _geometry(args):
    depth, saved_k = load_depth(args.depth)
    h, w = depth.shape
    camera = Camera.from_json(args.camera) if args.camera else None
    if camera is not None:
        camera.check_size((w, h))
    if args.rectified and camera is not None:
        raise ValueError('--rectified and --camera are alternatives: the camera transforms raw boxes/masks')
    k = saved_k if saved_k is not None else camera.matrix((w, h)) if camera is not None else None
    if k is None:
        raise ValueError('NPY depth requires --camera; rectified mode requires NPZ containing depth and k')
    calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else None
    records = load_boxes(args.boxes, depth.shape)
    measured = measure_records(depth, k, records, camera, calibration, args.seed, args.mask_area)
    image = read_image(args.image) if args.image else None
    if image is not None and image.shape[:2] != depth.shape:
        raise ValueError('image and depth must have the same spatial shape')
    write_frame(args.output, Path(args.depth).stem, image, measured,
                [record.get('_mask') for record in records], {'mode': 'geometry-only', 'seed': args.seed})
    print(f'{len(measured)} boxes -> {args.output}')


def _calibrate(args):
    calibration = fit_calibration(read_records(args.input))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(calibration, indent=2)+'\n', encoding='utf-8')
    print(f'training-only calibration -> {target}')


def _demo(args):
    """An analytic flat-plane fixture, independent of paper results or weights."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    h, w = 480, 640
    k = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
    depth = np.full((h, w), 4., np.float32)
    image = np.full((h, w, 3), (43, 35, 29), np.uint8)
    for x in range(0, w, 40):
        cv2.line(image, (x, 0), (x, h-1), (65, 56, 49), 1)
    for y in range(0, h, 40):
        cv2.line(image, (0, y), (w-1, y), (65, 56, 49), 1)
    cv2.ellipse(image, (320, 240), (86, 58), -8, 0, 360, (18, 16, 14), -1)
    cv2.putText(image, 'SYNTHETIC PLANE / NO MODEL WEIGHTS', (24, 36), cv2.FONT_HERSHEY_SIMPLEX, .65, (180, 200, 210), 1, cv2.LINE_AA)
    np.savez_compressed(output/'depth.npz', depth=depth, k=k)
    cv2.imwrite(str(output/'input.png'), image)
    boxes = [{'box': [220, 160, 420, 320], 'confidence': None}]
    (output/'boxes.json').write_text(json.dumps(boxes, indent=2)+'\n')
    measured = measure_records(depth, k, boxes, camera=None, seed=1130)
    write_frame(output, 'synthetic', image, measured, metadata={
        'mode': 'synthetic analytic fixture; not a learned-model result',
        'expected_bbox_footprint': 2.048, 'depth_units': 'metres',
        'calibration': None,
    })
    value = measured[0]['geometry']['bbox_raw']
    if not np.isclose(value, 2.048, atol=1e-10, rtol=0):
        raise RuntimeError(f'analytic plane check failed: {value}')
    print(f'CPU demo passed: raw enclosing footprint {value:.6f} m^2; no calibrated pothole area. Files: {output}')


def parser():
    command = argparse.ArgumentParser(prog='pothole-measure', description='ACSH recognition to local-road-plane footprint measurement')
    sub = command.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='run a CPU analytic scene with no model weights')
    demo.add_argument('--output', default='outputs/demo')
    demo.set_defaults(func=_demo)

    geometry = sub.add_parser('geometry', help='measure saved boxes and rectified depth without neural models')
    geometry.add_argument('--depth', required=True, help='NPY depth or NPZ with depth and optional rectified k')
    geometry.add_argument('--boxes', required=True, help='JSON xyxy arrays or {box, confidence?, mask?} records')
    coordinates = geometry.add_mutually_exclusive_group(required=True)
    coordinates.add_argument('--camera', help='camera JSON: boxes and optional masks are in the original image grid')
    coordinates.add_argument('--rectified', action='store_true', help='boxes/masks already rectified; depth NPZ must include k')
    geometry.add_argument('--image', help='optional image in the same coordinate system as boxes for overlays')
    geometry.add_argument('--calibration', help='frozen training-only log-OLS coefficient JSON')
    geometry.add_argument('--mask-area', action='store_true', help='also compute the auxiliary mask-area comparison')
    geometry.add_argument('--seed', type=int, default=1130)
    geometry.add_argument('--output', required=True, help='directory for JSON and optional overlay')
    geometry.set_defaults(func=_geometry)

    calibrate = sub.add_parser('calibrate', help='fit route coefficients using explicit split=train records only')
    calibrate.add_argument('--input', required=True, help='CSV, JSON, or JSONL containing reference_m2 and geometry fields')
    calibrate.add_argument('--output', required=True, help='coefficient JSON file')
    calibrate.set_defaults(func=_calibrate)

    infer = sub.add_parser('infer', help='predict boxes/masks from local ACSH weights and measure box footprints')
    infer.add_argument('--weights', required=True, help='local trusted ACSH or YOLO checkpoint')
    infer.add_argument('--source', required=True, help='image, image directory, or video')
    infer.add_argument('--camera', required=True, help='camera JSON matching the image resolution')
    depth = infer.add_mutually_exclusive_group(required=True)
    depth.add_argument('--depth', help='rectified NPY/NPZ for one image, or directory of frame-stem.npz caches')
    depth.add_argument('--depth-model', help='local Hugging Face metric-depth model directory; no automatic downloads')
    infer.add_argument('--calibration', help='frozen training-only coefficients; omit to export raw footprints only')
    infer.add_argument('--device', default='cpu')
    infer.add_argument('--threads', type=int, default=4, help='CPU intra-op and OpenCV thread limit')
    infer.add_argument('--imgsz', type=int, default=768)
    infer.add_argument('--conf', type=float, default=.05)
    infer.add_argument('--iou', type=float, default=.7)
    infer.add_argument('--seed', type=int, default=1130)
    infer.add_argument('--mask-area', action='store_true', help='also evaluate mask projection; paper main path uses boxes')
    infer.add_argument('--max-frames', type=int, help='stop after this many source frames')
    infer.add_argument('--frame-stride', type=int, default=1)
    infer.add_argument('--output', required=True)
    infer.set_defaults(func=lambda args: __import__('pothole_measurement.inference', fromlist=['run_inference']).run_inference(args))

    train = sub.add_parser('train', help='train the ACSH segmentation architecture from a local YOLOv8n-seg checkpoint')
    train.add_argument('--data', required=True, help='local Ultralytics segmentation data.yaml')
    train.add_argument('--weights', required=True, help='local YOLOv8n-seg pretrained checkpoint')
    train.add_argument('--config', required=True, help='architecture/recipe JSON; see configs/acsh-segmentation.json')
    train.add_argument('--device', default='cpu')
    train.add_argument('--epochs', type=int, help='override the original 60-epoch budget')
    train.add_argument('--seed', type=int)
    train.add_argument('--name', help='output run name override')
    train.add_argument('--batch', type=int, default=8)
    train.add_argument('--workers', type=int, default=4)
    train.add_argument('--threads', type=int, default=4)
    train.add_argument('--output', required=True, help='training project directory')
    train.set_defaults(func=lambda args: __import__('pothole_measurement.training', fromlist=['train']).train(args))
    return command


def main():
    command = parser()
    args = command.parse_args()
    if getattr(args, 'threads', 1) < 1:
        command.error('threads must be positive')
    if getattr(args, 'frame_stride', 1) < 1 or (getattr(args, 'max_frames', None) is not None and args.max_frames < 1):
        command.error('frame-stride and max-frames must be positive')
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as error:
        command.error(str(error))
