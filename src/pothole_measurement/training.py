"""The authors' ACSH segmentation training recipe with explicit local inputs."""
from __future__ import annotations
import json
from pathlib import Path


def train(args):
    import torch
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from .models import load_model
    from .models import refine_architecture
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config.get('task') != 'segment':
        raise ValueError('this entrypoint supports the adopted segmentation architecture')
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.seed is not None:
        config['seed'] = args.seed
    if args.name is not None:
        config['name'] = args.name
    data = Path(args.data).expanduser().resolve(strict=True)
    weights = Path(args.weights).expanduser().resolve(strict=True)
    project = Path(args.output).expanduser().resolve()
    torch.set_num_threads(args.threads)

    class ACSHTrainer(SegmentationTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return refine_architecture.build(config, weights)

        def validate(self):
            # Selection uses full-precision mask AP, matching experiment.py.
            amp = self.amp
            self.amp = False
            try:
                metrics = self.validator(self)
            finally:
                self.amp = amp
            if metrics is None:
                return None, None
            metrics.pop('fitness', None)
            fitness = metrics['metrics/mAP50-95(M)']
            if self.best_fitness is None or self.best_fitness < fitness:
                self.best_fitness = fitness
            return metrics, fitness

    model = load_model(weights)
    if model.task != 'segment':
        raise ValueError('--weights must be local YOLOv8n-seg pretrained weights')
    kwargs = dict(data=str(data), epochs=config['epochs'], imgsz=config['imgsz'], batch=args.batch,
                  device=args.device, optimizer='AdamW', lr0=.0007, lrf=.03,
                  momentum=.9, weight_decay=.0005, warmup_epochs=3, warmup_bias_lr=0.,
                  cos_lr=True, nbs=16, hsv_h=.01, hsv_s=.3, hsv_v=.3, scale=.2,
                  translate=.05, fliplr=.5, mosaic=.2, close_mosaic=15, mixup=0.,
                  copy_paste=0., degrees=0., shear=0., perspective=0., erasing=0.,
                  auto_augment=None, workers=args.workers, seed=config['seed'], deterministic=True,
                  amp=True, plots=False, patience=config['epochs']+1, project=str(project),
                  name=config['name'], exist_ok=False, save=True, verbose=False, mask_ratio=2)
    # Preserve the historical recipe; upstream AMP checks may request a
    # comparison model when a user chooses to run training.
    kwargs.update(config.get('recipe', {}))
    def save_config(trainer):
        (trainer.save_dir/'experiment.json').write_text(json.dumps(config, indent=2)+'\n')
    model.add_callback('on_train_start', save_config)
    return model.train(trainer=ACSHTrainer, **kwargs)
