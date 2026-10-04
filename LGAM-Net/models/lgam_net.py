import torch
from torch import nn
from torch.nn import functional as F
from .encoders import CLIPEncoder, TinyEncoder
from .ra_lfe import RALFE

def compare(a, b):
    return torch.cat((a + b, (a - b).abs(), a * b), dim=-1)

class LocalAlignment(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.visual = nn.Linear(dim, dim)
        self.text = nn.Linear(dim, dim)
        self.output = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, visual, text, valid):
        v, t = self.visual(visual), self.text(text)
        scores = (v @ t.transpose(1, 2)) / v.shape[-1] ** 0.5
        scores = scores.masked_fill(~valid[:, None, :], torch.finfo(scores.dtype).min)
        text_at_visual = scores.softmax(-1) @ t
        visual_at_text = scores.transpose(1, 2).softmax(-1) @ v
        dv = (v - text_at_visual).abs().mean(1)
        dt = ((t - visual_at_text).abs() * valid[..., None]).sum(1)
        dt = dt / valid.sum(1, keepdim=True).clamp_min(1)
        return self.output(torch.cat((dv, dt), -1))

class LGAMNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        self.encoder = TinyEncoder() if c.get('encoder') == 'tiny' else CLIPEncoder(
            c['clip_name'], c.get('freeze_encoder', True), c.get('local_files_only', False))
        e, d = self.encoder, c['dim']
        self.visual_project = nn.Linear(e.visual_dim, d)
        self.text_project = nn.Linear(e.text_dim, d)
        self.image_global = nn.Linear(e.global_dim, d)
        self.text_global = nn.Linear(e.global_dim, d)
        self.enhance = RALFE(d, c.get('views', 4), c.get('use_ra', True), c.get('use_lfe', True))
        self.local = LocalAlignment(d)
        self.use_local = c.get('use_local_alignment', True)
        self.global_compare = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.LayerNorm(d))
        self.pair = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.head = nn.Sequential(nn.Linear(6 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1))

    def forward(self, batch):
        b = batch['pixels'].shape[0]
        # [all caption1, all caption2] for one shared text encoder call.
        ids = torch.cat((batch['ids1'], batch['ids2']))
        masks = torch.cat((batch['mask1'], batch['mask2']))
        p, ig, tok, tg, grid = self.encoder(batch['pixels'], ids, masks)
        visual = self.enhance(self.visual_project(p), grid)
        ig = self.image_global(ig)
        text = self.text_project(tok)
        tg = self.text_global(tg)
        pair_features = []
        for k in range(2):
            local = self.local(visual, text[k*b:(k+1)*b], batch[f'local_mask{k+1}']) if self.use_local else torch.zeros_like(ig)
            glob = self.global_compare(compare(ig, tg[k*b:(k+1)*b]))
            pair_features.append(self.pair(torch.cat((local, glob), -1)))
        fused = torch.cat((compare(*pair_features), compare(tg[:b], tg[b:])), -1)
        return {'logits': self.head(fused).squeeze(-1),
                'image_embeddings': F.normalize(ig, dim=-1),
                'text_embeddings': F.normalize(tg[:b], dim=-1)}
