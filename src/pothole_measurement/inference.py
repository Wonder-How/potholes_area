"""Local-checkpoint inference and portable JSON/overlay exports."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from .calibration import calibrated_area
from .geometry import geometry

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'cannot decode image: {path}')
    return image


def load_depth(path):
    """Read NPY depth or NPZ with depth and optional rectified k."""
    path = Path(path)
    if path.suffix.lower() == '.npy':
        depth, k = np.load(path, allow_pickle=False), None
    else:
        with np.load(path, allow_pickle=False) as saved:
            depth = saved['depth'].copy()
            k = saved['k'].copy() if 'k' in saved else None
    if depth.ndim != 2:
        raise ValueError('depth must be a two-dimensional rectified-grid array')
    return depth, k


def load_boxes(path, shape):
    """Load xyxy arrays or records {box, confidence?, mask?}; masks are image paths."""
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict):
        data = data['predictions']
    records = []
    for item in data:
        record = dict(item) if isinstance(item, dict) else {'box': item}
        record.setdefault('confidence', None)
        record.pop('geometry', None)
        if record.get('mask') is not None:
            mask_path = Path(record.pop('mask'))
            if not mask_path.is_absolute():
                mask_path = path.parent / mask_path
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != shape:
                raise ValueError(f'mask must decode to depth size {shape}: {mask_path}')
            record['_mask'] = (mask > 0).astype(np.uint8)
        records.append(record)
    return records


def measure_records(depth, k, records, camera, calibration=None, seed=1130, mask_area=False):
    measured = []
    for index, original in enumerate(records):
        record = {key: value for key, value in original.items() if not key.startswith('_')}
        mask = original.get('_mask')
        geo = geometry(depth, record['box'], k, seed, mask if mask_area else None, camera=camera)
        record.update(prediction_index=index, geometry=geo,
                      bbox_m2=calibrated_area(geo, calibration, 'bbox'),
                      mask_m2=calibrated_area(geo, calibration, 'mask') if mask_area else None)
        measured.append(record)
    return measured


def overlay(image, records, masks=None):
    """Render on the original image grid; measurement is from predicted boxes."""
    image = image.copy()
    font_scale = max(.45, min(image.shape[:2])/1200)
    line_width = max(1, round(min(image.shape[:2])/500))
    for index, record in enumerate(records):
        color = (83, 224, 155) if record['geometry']['bbox_valid'] else (80, 130, 245)
        if masks is not None and index < len(masks) and masks[index] is not None:
            selected = masks[index].astype(bool)
            image[selected] = (.72*image[selected]+.28*np.asarray(color)).astype(np.uint8)
        x0, y0, x1, y1 = np.rint(record['box']).astype(int)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, line_width)
        area = record['bbox_m2']
        if area is not None:
            text = f'#{index+1}  {area:.3f} m^2'
        elif record['geometry']['bbox_valid']:
            text = f'#{index+1}  footprint {record["geometry"]["bbox_raw"]:.3f} (uncalibrated)'
        else:
            text = f'#{index+1}  invalid depth support'
        confidence = record.get('confidence')
        if confidence is not None:
            text += f'  {confidence:.2f}'
        position = (max(0, x0), max(18, y0-8))
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (12, 19, 27), line_width+3, cv2.LINE_AA)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, line_width, cv2.LINE_AA)
    return image


def write_frame(output, name, image, records, masks=None, metadata=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = {'sample': name, 'adopted_area_path': 'segmentation-predicted bbox',
             'raw_area_units': 'squared depth units; uncalibrated enclosing footprint',
             'predictions': records, **(metadata or {})}
    target = output / f'{name}.json'
    target.write_text(json.dumps(frame, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    if image is not None:
        if not cv2.imwrite(str(output/f'{name}.jpg'), overlay(image, records, masks)):
            raise RuntimeError(f'cannot write overlay for {name}')
    return frame


class LocalMetricDepth:
    def __init__(self, model_path, device='cpu'):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        model_path = Path(model_path).expanduser().resolve(strict=True)
        if not model_path.is_dir():
            raise ValueError('--depth-model must be a local Hugging Face model directory')
        self.torch = torch
        self.device = f"cuda:{device}" if str(device).isdigit() else device
        self.processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True, use_fast=False)
        self.model = AutoModelForDepthEstimation.from_pretrained(str(model_path), local_files_only=True).to(self.device).eval()

    def __call__(self, rectified_bgr):
        torch = self.torch
        inputs = self.processor(images=cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB), return_tensors='pt')
        with torch.inference_mode():
            output = self.model(**{key: value.to(self.device) for key, value in inputs.items()})
            depth = torch.nn.functional.interpolate(output.predicted_depth[0][None, None], size=rectified_bgr.shape[:2], mode='bicubic', align_corners=False)[0, 0]
        return depth.float().cpu().numpy()


def frames_from_source(source, max_frames=None, frame_stride=1):
    source = Path(source).expanduser().resolve(strict=True)
    if source.is_dir():
        paths = sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise ValueError('source directory has no supported images')
        if len({path.stem for path in paths}) != len(paths):
            raise ValueError('source image basenames must be unique for JSON exports')
        for index, path in enumerate(paths):
            if index % frame_stride == 0:
                yield path.stem, read_image(path)
                if max_frames is not None:
                    max_frames -= 1
                    if max_frames <= 0:
                        break
    elif source.suffix.lower() in IMAGE_SUFFIXES:
        yield source.stem, read_image(source)
    else:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f'cannot open video: {source}')
        try:
            index, written = 0, 0
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if index % frame_stride == 0:
                    yield f'{source.stem}_{index:06d}', image
                    written += 1
                    if max_frames is not None and written >= max_frames:
                        break
                index += 1
        finally:
            capture.release()


def run_inference(args):
    import torch
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    from .camera import Camera
    from .models import load_model
    camera = Camera.from_json(args.camera)
    calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else None
    source = Path(args.source).expanduser().resolve(strict=True)
    depth_path = Path(args.depth).expanduser().resolve(strict=True) if args.depth else None
    if depth_path is not None and depth_path.is_file() and (source.is_dir() or source.suffix.lower() not in IMAGE_SUFFIXES):
        raise ValueError('multiple frames require --depth directory with one stem.npz per frame, or --depth-model')
    model = load_model(args.weights)
    if model.task not in ('detect', 'segment'):
        raise ValueError('--weights must be an object detection or instance segmentation checkpoint')
    depth_model = LocalMetricDepth(args.depth_model, args.device) if args.depth_model else None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    with (output/'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for name, image in frames_from_source(source, args.max_frames, args.frame_stride):
            h, w = image.shape[:2]
            camera.check_size((w, h))
            k = camera.matrix((w, h))
            if depth_path is not None:
                cache = depth_path/f'{name}.npz' if depth_path.is_dir() else depth_path
                depth, saved_k = load_depth(cache)
                if saved_k is not None:
                    k = saved_k
            else:
                depth = depth_model(camera.rectify_image(image, k))
            if depth.shape != (h, w):
                raise ValueError(f'depth shape {depth.shape} differs from raw/rectified image shape {(h,w)}')
            result = model.predict(image, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                   max_det=300, device=args.device, half=False,
                                   retina_masks=True, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            masks = None if result.masks is None else result.masks.data.cpu().numpy().astype(np.uint8)
            if masks is not None and tuple(masks.shape[1:]) != depth.shape:
                raise ValueError('native decoded masks do not match the input grid')
            records = [{'box': box.tolist(), 'confidence': float(score),
                        '_mask': masks[index] if masks is not None else None}
                       for index, (box, score) in enumerate(zip(boxes, scores))]
            records = measure_records(depth, k, records, camera, calibration, args.seed, args.mask_area)
            frame = write_frame(output, name, image, records, masks,
                                {'camera_file': str(args.camera), 'seed': args.seed,
                                 'mask_area_comparison_enabled': args.mask_area})
            stream.write(json.dumps(frame, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            count += 1
            print(f'{name}: {len(records)} predictions -> {output}', flush=True)
    if count == 0:
        raise ValueError('source yielded no frames')
