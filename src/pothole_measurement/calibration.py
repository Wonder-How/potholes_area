"""Training-only, route-specific log ordinary least squares calibration."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np


def read_records(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))
    if path.suffix.lower() == '.jsonl':
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, list):
        raise ValueError('calibration input JSON must be a list of records')
    return result


def _valid(value):
    return value is True or str(value).lower() in ('true', '1')


def fit_calibration(records):
    """Fit only explicit split=train rows with valid positive raw/reference area.

    Record fields: split, reference_m2, bbox_valid, bbox_route, bbox_raw;
    optional mask_valid, mask_route, mask_raw. Missing routes remain unavailable.
    """
    training = [record for record in records if record.get('split') == 'train']
    if not training:
        raise ValueError('no explicit split=train records; refusing to calibrate on another partition')
    coefficients = {}
    for kind in ('bbox', 'mask'):
        for route in ('plane', 'fallback'):
            pairs = []
            for record in training:
                geometry = record.get('geometry', record)
                if geometry.get(kind+'_route') != route or not _valid(geometry.get(kind+'_valid', False)):
                    continue
                raw, reference = float(geometry[kind+'_raw']), float(record['reference_m2'])
                if np.isfinite([raw, reference]).all() and min(raw, reference) > 0:
                    pairs.append((raw, reference))
            if len(pairs) < 2:
                continue
            values = np.log(np.asarray(pairs))
            design = np.c_[np.ones(len(values)), values[:, 0]]
            if np.linalg.matrix_rank(design) < 2:
                raise ValueError(f'{kind}_{route} needs at least two distinct raw areas')
            name = f'{kind}_{route}'
            coefficients[name] = np.linalg.lstsq(design, values[:, 1], rcond=None)[0].tolist()
            coefficients[name+'_train_n'] = len(pairs)
    if not coefficients:
        raise ValueError('no route has two valid positive training observations')
    coefficients['_metadata'] = {
        'method': 'log ordinary least squares per geometry route',
        'fit_split': 'train', 'training_records': len(training),
        'ignored_nontraining_records': len(records)-len(training),
    }
    return coefficients


def calibrated_area(geometry, coefficients, kind='bbox'):
    """Return calibrated m² or None; missing coefficients never become identity."""
    if coefficients is None or not geometry.get(kind+'_valid'):
        return None
    raw = geometry.get(kind+'_raw')
    name = f"{kind}_{geometry.get(kind+'_route')}"
    if raw is None or not np.isfinite(raw) or raw <= 0 or name not in coefficients:
        return None
    coefficient = np.asarray(coefficients[name], dtype=float)
    if coefficient.shape != (2,) or not np.isfinite(coefficient).all():
        raise ValueError(f'{name} calibration must contain finite [intercept, slope]')
    with np.errstate(over='ignore', invalid='ignore'):
        value = float(np.exp(coefficient[0]+coefficient[1]*np.log(raw)))
    return value if np.isfinite(value) and value > 0 else None
