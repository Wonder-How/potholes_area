"""Keep P2 predictions while decoupling their mask prototype input.

All pretrained source tensors are transferred by base_architecture, unchanged.
P2 remains supervised at stride4; prototypes always remain stride4 as well.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules.head import Detect, Segment
from ultralytics.utils.loss import v8SegmentationLoss

from .base_architecture import build as base_build, digest
from .acmix_modules import IdentityBlendACmix


class DetailFusion(nn.Module):
    def __init__(self, channels, gate=.1):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.dirac_(self.proj.weight)
        self.gate = nn.Parameter(torch.tensor(float(gate)))

    def forward(self, p3, p2):
        # Both tensors enter Proto on the original P3 spatial grid.
        detail = self.proj(F.avg_pool2d(p2, 2))
        return p3 + self.gate.tanh() * detail


class StableP2Feature(nn.Module):
    """Learn fine P2 fusion from an upsampled pretrained P3 starting point."""
    def __init__(self, base, channels, gate=.1):
        super().__init__()
        self.base = base
        self.channels = channels
        self.gate = nn.Parameter(torch.tensor(float(gate)))
        for attr in ('i','f','type','np'):
            if hasattr(base,attr):setattr(self,attr,getattr(base,attr))
        self.type='StableP2Feature'

    def forward(self, x):
        p3 = x[:, :self.channels]
        return p3 + self.gate * (self.base(x) - p3)


class DecoupledSegment(Segment):
    def forward(self, x):
        # Detect.forward invokes Segment.forward_head, preserving all four
        # supervised coefficient branches and the native output dictionary.
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto_input = x[self.proto_p3_index]
        if self.detail_fusion is not None:
            proto_input = self.detail_fusion(proto_input, x[self.proto_p2_index])
        proto = self.proto(self.proto_adapter(proto_input))
        if getattr(self, 'output_detail', None) is not None:
            # Direct stride4 detail bypasses pool-to-stride8/re-upsampling.
            # The common prototype grid and evaluator remain unchanged.
            proto = proto + .1 * self.output_detail(x[self.proto_p2_index])
        if isinstance(preds, dict):
            if 'one2one' in preds:
                preds['one2many']['proto'] = proto
                preds['one2one']['proto'] = proto.detach()
            else:
                preds['proto'] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class MaskCriterion(v8SegmentationLoss):
    """Optional crop-aware Dice supplement, shared by every factorial cell."""
    def __init__(self, model):
        super().__init__(model)
        self.dice_weight = float(model.refine_config.get('dice_weight', 0))
        self.mask_gain = float(model.refine_config.get('mask_gain', 1))

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        from ultralytics.utils.ops import crop_mask
        logits = torch.einsum('in,nhw->ihw', pred, proto)
        bce = F.binary_cross_entropy_with_logits(logits, gt_mask, reduction='none')
        result = (crop_mask(bce, xyxy).mean(dim=(1, 2)) / area).sum()
        if self.dice_weight:
            # Native crop_mask uses in-place slicing for small CUDA batches.
            # Preserve sigmoid's saved output and BCE's target for backward.
            probability = crop_mask(logits.sigmoid().clone(), xyxy)
            target = crop_mask(gt_mask.clone(), xyxy)
            intersection = (probability * target).sum((1, 2))
            denominator = probability.sum((1, 2)) + target.sum((1, 2))
            result = result + self.dice_weight * (1 - (2*intersection+1) / (denominator+1)).sum()
        return result * self.mask_gain


def init_criterion(self):
    return MaskCriterion(self)


def build(config, weights):
    from types import MethodType
    p2 = config['p2']
    mode = config.get('p2_mode', 'parallel')
    model = base_build('segment', p2, {}, weights, config['seed'], mode)
    head = model.model[-1]
    # PAN had a pooled P2 -> Proto wrapper; recover its unchanged native Proto.
    if p2 and mode == 'pan':
        head.proto = head.proto[1]
        mapping = model.transfer_audit['tensor_mapping']
        model.transfer_audit['tensor_mapping'] = [
            (a, b.replace('model.28.proto.1.', 'model.28.proto.')) for a, b in mapping]
    head.__class__ = DecoupledSegment
    head.proto_p3_index = 1 if p2 and mode == 'pan' else 0
    head.proto_p2_index = (0 if mode == 'pan' else 3) if p2 else None
    channels = head.proto.cv1.conv.in_channels
    head.detail_fusion = DetailFusion(channels, config.get('detail_gate', .1)) if p2 and config.get('detail', False) else None
    identity = nn.Identity()
    identity.type = 'Identity'
    head.proto_adapter = (IdentityBlendACmix(identity, channels,
                              config.get('kernel', 3), config.get('gate', .1))
                          if config['acmix'] else nn.Identity())
    head.output_detail = None
    if p2 and config.get('output_detail', False):
        head.output_detail = nn.Conv2d(channels, head.nm, 1)
        nn.init.normal_(head.output_detail.weight,std=.01)
        nn.init.zeros_(head.output_detail.bias)
    if p2 and config.get('stable_p2', False):
        assert mode=='parallel'
        model.model[24] = StableP2Feature(model.model[24],channels,config.get('p2_gate',.1))
    if p2 and config.get('scale_prior', False):
        # Only the *additional* P2 classifier changes; every original source
        # tensor remains exactly preserved. Copying P3 had given four times
        # as many anchors the same per-location foreground prior.
        with torch.no_grad():
            head.cv3[head.proto_p2_index][-1].bias.sub_(math.log(4))
    model.refine_config = dict(config)
    # A method on the class is serialized as part of the actual model class,
    # rather than a bound per-instance function that breaks checkpoint loads.
    from ultralytics.nn.tasks import SegmentationModel
    model.__class__ = RefinedSegmentationModel
    model.initial_architecture_sha256 = digest(model.state_dict())
    model.transfer_audit.update(proto_input='P3', prototype_stride=4,
                                acmix_site='prototype_only' if config['acmix'] else None)
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(config['seed'], deterministic=True)
    return model


from ultralytics.nn.tasks import SegmentationModel


class RefinedSegmentationModel(SegmentationModel):
    def init_criterion(self):
        return MaskCriterion(self)
