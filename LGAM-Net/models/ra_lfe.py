"""Linear-in-spatial-size residual context and local feature enhancement."""
import torch
from torch import nn

class RALFE(nn.Module):
    def __init__(self, dim, views=4, use_ra=True, use_lfe=True):
        super().__init__()
        if dim % views:
            raise ValueError('dim must be divisible by views')
        self.views, self.q = views, dim // views
        self.use_ra, self.use_lfe = use_ra, use_lfe
        self.norm = nn.LayerNorm(dim)
        self.project = nn.Conv2d(dim, dim, 1)
        self.qkv = nn.Linear(self.q, 3 * self.q)
        self.view_out = nn.Linear(self.q, self.q)
        self.depthwise = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.local_weights = nn.Conv2d(dim, views, 1)
        self.context_score = nn.Conv2d(dim, 1, 1)
        self.context_weights = nn.Linear(dim, views)
        self.context_value = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gate = nn.Conv2d(dim, 1, 1)
        self.output = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, grid):
        if not self.use_ra and not self.use_lfe:
            return x
        b, n, d = x.shape
        h, w = grid
        if n != h * w:
            raise ValueError('Patch sequence/grid mismatch')
        z = self.project(self.norm(x).transpose(1, 2).reshape(b, d, h, w))
        local = torch.nn.functional.gelu(self.depthwise(z))
        context = None
        if self.use_ra:
            attention = self.context_score(z).flatten(2).softmax(-1)
            context = (z.flatten(2) * attention).sum(-1)
        if self.use_lfe:
            s = z.flatten(2).transpose(1, 2).reshape(b, n, self.views, self.q)
            q, k, v = self.qkv(s).chunk(3, dim=-1)
            a = ((q @ k.transpose(-1, -2)) / self.q ** 0.5).softmax(-1)
            e = s + self.view_out(a @ v)
            weights = self.local_weights(local).flatten(2).transpose(1, 2)
            if context is not None:
                weights = weights + self.context_weights(context)[:, None, :]
            fused = (e * weights.softmax(-1).unsqueeze(-1)).reshape(b, n, d)
            u = fused.transpose(1, 2).reshape(b, d, h, w)
        else:
            u = torch.zeros_like(z)
        if context is not None:
            u = u + self.gate(local).sigmoid() * self.context_value(context)[:, :, None, None]
        return x + self.gamma * self.output(u).flatten(2).transpose(1, 2)
