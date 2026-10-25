"""Residual ACmix adapters for YOLOv8, without changing pretrained layer indices.

ACmix equations follow Pan et al., CVPR 2022, and the supplied reproduction
implementation. Device placement, configurable windows, and residual adaptation
are local changes. The original reproduction package is left untouched.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ACmix(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, heads: int = 4):
        super().__init__()
        assert channels % heads == 0 and kernel % 2 == 1
        self.channels, self.heads, self.dim, self.kernel = channels, heads, channels // heads, kernel
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.pos = nn.Conv2d(2, self.dim, 1)
        self.fc = nn.Conv2d(3 * heads, 9, 1, bias=False)
        self.dep_conv = nn.Conv2d(9 * self.dim, channels, 3, padding=1, groups=self.dim, bias=False)
        self.rate_att = nn.Parameter(torch.tensor(0.5))
        self.rate_conv = nn.Parameter(torch.tensor(0.5))
        with torch.no_grad():
            self.dep_conv.weight.copy_(torch.eye(9).reshape(9, 3, 3).repeat(channels, 1, 1, 1))

    def initialize_identity(self):
        """Initialize the dynamic-convolution path to exact identity, not zero.

        Encode individual value heads into four aggregation channels and select
        each at the center tap. The value permutation compensates the grouped
        convolution's interleaved output-channel order.
        """
        assert self.heads <= 9
        with torch.no_grad():
            for layer in (self.q, self.k, self.v):
                layer.weight.zero_(); layer.bias.zero_()
                for d in range(self.dim):
                    for h in range(self.heads):
                        layer.weight[h*self.dim+d, d*self.heads+h, 0, 0] = 1
            self.pos.weight.zero_(); self.pos.bias.zero_()
            self.fc.weight.zero_(); self.dep_conv.weight.zero_()
            for h in range(self.heads):
                self.fc.weight[h, 2*self.heads+h, 0, 0] = 1
            for c in range(self.channels):
                self.dep_conv.weight[c, c % self.heads, 1, 1] = 1
            self.rate_att.zero_(); self.rate_conv.fill_(1)
        self.unpermute_attention = True

    def forward(self, x):
        b, _, h, w = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
                                torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype), indexing='ij')
        pe = self.pos(torch.stack((xx, yy), 0).unsqueeze(0))
        qa = q.reshape(b * self.heads, self.dim, h, w) * self.dim ** -0.5
        ka, va = (z.reshape(b * self.heads, self.dim, h, w) for z in (k, v))
        pad = self.kernel // 2
        def unfold(z):
            return F.unfold(F.pad(z, (pad, pad, pad, pad), mode='reflect'), self.kernel).reshape(
                z.shape[0], self.dim, self.kernel ** 2, h, w)
        att = (qa.unsqueeze(2) * (unfold(ka) + pe.unsqueeze(2) - unfold(pe))).sum(1).softmax(1)
        a = (att.unsqueeze(1) * unfold(va)).sum(2).reshape(b, self.channels, h, w)
        if getattr(self, 'unpermute_attention', False):
            a = a.reshape(b, self.heads, self.dim, h, w).permute(0, 2, 1, 3, 4).reshape(b, self.channels, h, w)
        f = self.fc(torch.cat([z.reshape(b, self.heads, self.dim, h*w) for z in (q, k, v)], 1))
        c = self.dep_conv(f.permute(0, 2, 1, 3).reshape(b, -1, h, w))
        return self.rate_att * a + self.rate_conv * c


class ResidualACmix(nn.Module):
    def __init__(self, base: nn.Module, channels: int, kernel: int, gate: float, norm_scale: float = 1.0,
                 norm_eps: float = 1e-5, norm_momentum: float = 0.1):
        super().__init__()
        self.base = base
        self.acmix = ACmix(channels, kernel)
        self.norm = nn.BatchNorm2d(channels, eps=norm_eps, momentum=norm_momentum)
        nn.init.constant_(self.norm.weight, norm_scale)
        self.gate = nn.Parameter(torch.tensor(gate))
        for attr in ('i', 'f', 'type', 'np'):
            if hasattr(base, attr):
                setattr(self, attr, getattr(base, attr))
        self.type = 'ResidualACmix(' + getattr(base, 'type', type(base).__name__) + ')'

    def forward(self, x):
        x = self.base(x)
        return x + self.gate * self.norm(self.acmix(x))


class IdentityBlendACmix(ResidualACmix):
    """Function-preserving ACmix whose convolution parameters learn immediately."""
    def __init__(self, base, channels, kernel, gate):
        nn.Module.__init__(self)
        self.base = base
        self.acmix = ACmix(channels, kernel)
        self.acmix.initialize_identity()
        self.norm = nn.Identity()
        self.gate = nn.Parameter(torch.tensor(gate))
        for attr in ('i', 'f', 'type', 'np'):
            if hasattr(base, attr):setattr(self, attr, getattr(base, attr))
        self.type = 'IdentityBlendACmix(' + getattr(base, 'type', type(base).__name__) + ')'

    def forward(self, x):
        x = self.base(x)
        return x + self.gate * (self.acmix(x) - x)


def attach_acmix(model, config):
    indices = config.get('layers', [])
    for idx in indices:
        base = model.model[idx]
        channels = base.cv2.conv.out_channels
        if config.get('initialization') == 'dirac':
            model.model[idx] = IdentityBlendACmix(base, channels, config.get('kernel',3), config.get('gate',.1))
        else:
            model.model[idx] = ResidualACmix(base, channels, config.get('kernel', 3), config.get('gate', 0.05),
                                           config.get('norm_scale', 1.0), config.get('norm_eps', 1e-5),
                                           config.get('norm_momentum', 0.1))
    model.acmix_config = config
    # Module construction consumes RNG. Restore the same augmentation RNG stream
    # for paired baseline and adapter runs in the trainer after construction.
    return model
