"""P2/P3/P4/P5 YOLOv8 with semantic, rather than index-only, transfer."""
import copy
import hashlib

import torch
from ultralytics.nn.tasks import DetectionModel, SegmentationModel
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import init_seeds
from ultralytics import __file__ as ultralytics_file
from pathlib import Path

from .acmix_modules import attach_acmix


def digest(state):
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def model_config(task, p2=False, p2_mode='pan'):
    root = Path(ultralytics_file).parent / 'cfg/models/v8'
    cfg = YAML.load(root / ('yolov8-p2.yaml' if p2 and p2_mode=='pan' else
                            'yolov8-seg.yaml' if task == 'segment' else 'yolov8.yaml'))
    cfg['scale'] = 'n'
    cfg['nc'] = 1
    if p2 and p2_mode=='parallel':
        head=cfg['head'].pop()
        cfg['head'] += [[15,1,'nn.Upsample',[None,2,'nearest']],
                        [[-1,2],1,'Concat',[1]],[-1,3,'C2f',[256]]]
        head[0]=[15,18,21,24]
        cfg['head'].append(head)
    elif p2:
        # Keep the P2 feature width at 64 (nano) so that all existing head
        # branches and the segment prototype can transfer without resizing.
        cfg['head'][8][3][0] = 256   # global layer 18, P2 C2f
        cfg['head'][9][3][0] = 256   # global layer 19, P2 -> P3 Conv
        if task == 'segment':
            cfg['head'][-1][2:] = ['Segment', ['nc', 32, 256]]
    return cfg


def transfer_parallel(source,target):
    moved=[]
    for i in range(22):
        target.model[i].load_state_dict(source.model[i].state_dict())
        moved.extend((f'model.{i}.{k}',f'model.{i}.{k}') for k in source.model[i].state_dict())
    sh,th=source.model[-1],target.model[-1]
    for name in ('cv2','cv3','cv4'):
        if not hasattr(sh,name):continue
        for old,new in [(0,0),(1,1),(2,2),(0,3)]:
            getattr(th,name)[new].load_state_dict(getattr(sh,name)[old].state_dict())
        for i in range(3):
            moved.extend((f'model.22.{name}.{i}.{k}',f'model.25.{name}.{i}.{k}')
                         for k in getattr(sh,name)[i].state_dict())
    for name in ('dfl','proto'):
        if hasattr(sh,name):
            getattr(th,name).load_state_dict(getattr(sh,name).state_dict())
            moved.extend((f'model.22.{name}.{k}',f'model.25.{name}.{k}') for k in getattr(sh,name).state_dict())
    ts,ss=target.model[24].state_dict(),source.model[15].state_dict()
    target.model[24].load_state_dict({k:v for k,v in ss.items() if k in ts and v.shape==ts[k].shape},strict=False)
    ss,ts=source.state_dict(),target.state_dict()
    assert set(ss)=={a for a,b in moved}
    assert all(torch.equal(ss[a],ts[b]) for a,b in moved)
    return dict(mapped_tensors=len(moved),source_tensors=len(ss),all_source_tensors_preserved=True,
                tensor_mapping=moved,segmentation_prototype_stride=4 if hasattr(th,'proto') else None,
                p2_mode='parallel',original_neck_feature_paths_preserved=True)


def transfer_p2(source, target):
    """Preserve every standard-model tensor at its semantic destination.

    The extra P2 path is initialized from shape-compatible P3 tensors. Its
    incompatible input projections retain deterministic random initialization.
    The original three prediction branches become branches 1, 2, 3.
    """
    mapping = {**{i: i for i in range(16)}, 16: 22, 18: 24, 19: 25, 21: 27}
    moved = []
    for old, new in mapping.items():
        target.model[new].load_state_dict(source.model[old].state_dict(), strict=True)
        moved.extend((f'model.{old}.{k}', f'model.{new}.{k}')
                     for k in source.model[old].state_dict())
    sh, th = source.model[-1], target.model[-1]
    for name in ('cv2', 'cv3', 'cv4'):
        if not hasattr(sh, name):
            continue
        for old, new in [(0, 0), (0, 1), (1, 2), (2, 3)]:
            getattr(th, name)[new].load_state_dict(getattr(sh, name)[old].state_dict())
        for old in range(3):
            moved.extend((f'model.22.{name}.{old}.{k}', f'model.28.{name}.{old+1}.{k}')
                         for k in getattr(sh, name)[old].state_dict())
    for name in ('dfl', 'proto'):
        if hasattr(sh, name):
            getattr(th, name).load_state_dict(getattr(sh, name).state_dict())
            moved.extend((f'model.22.{name}.{k}', f'model.28.{name}.{k}')
                         for k in getattr(sh, name).state_dict())
    for new, old in [(18, 15), (19, 16), (21, 15)]:
        ts, ss = target.model[new].state_dict(), source.model[old].state_dict()
        compatible = {k: v for k, v in ss.items() if k in ts and v.shape == ts[k].shape}
        target.model[new].load_state_dict(compatible, strict=False)
    if hasattr(th, 'proto'):
        # Native YOLOv8 segmentation assumes stride-4 prototypes. Proto
        # upsamples by 2, so pool the stride-4 P2 input to stride 8 first.
        # P2 coefficients/detections still operate at stride 4.
        th.proto = torch.nn.Sequential(torch.nn.AvgPool2d(2), th.proto)
        moved = [(a, b.replace('model.28.proto.', 'model.28.proto.1.'))
                 for a,b in moved]
    ss, ts = source.state_dict(), target.state_dict()
    assert set(ss) == {a for a, b in moved}, set(ss) - {a for a, b in moved}
    assert all(torch.equal(ss[a], ts[b]) for a, b in moved)
    return dict(mapped_tensors=len(moved), source_tensors=len(ss),
                all_source_tensors_preserved=True, tensor_mapping=moved,
                segmentation_prototype_stride=4 if hasattr(th,'proto') else None)


def build(task, p2, adapter, weights, seed, p2_mode='pan'):
    init_seeds(seed, deterministic=True)
    cls = SegmentationModel if task == 'segment' else DetectionModel
    source = cls(model_config(task), nc=1, verbose=False)
    source.load(weights, verbose=False)
    shared_hash = digest(source.state_dict())
    if p2:
        model = cls(model_config(task, True, p2_mode), nc=1, verbose=False)
        transfer = (transfer_parallel if p2_mode=='parallel' else transfer_p2)(source, model)
    else:
        model = source
        transfer = dict(all_source_tensors_preserved=True, source_tensors=len(source.state_dict()))
    model.initial_shared_sha256 = shared_hash
    model.initial_architecture_sha256 = digest(model.state_dict())
    model.transfer_audit = transfer
    # Every factorial model uses exactly the same semantic ACmix site: the
    # top-down P3 neck feature, global layer 15. The P2 path consumes this output.
    attach_acmix(model, adapter)
    init_seeds(seed, deterministic=True)
    return model
