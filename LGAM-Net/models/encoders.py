import torch
from torch import nn

class CLIPEncoder(nn.Module):
    def __init__(self, name, freeze=True, local_files_only=False):
        super().__init__()
        from transformers import CLIPModel
        self.clip = CLIPModel.from_pretrained(name, local_files_only=local_files_only)
        self.freeze = freeze
        self.visual_dim = self.clip.config.vision_config.hidden_size
        self.text_dim = self.clip.config.text_config.hidden_size
        self.global_dim = self.clip.config.projection_dim
        self.patch_size = self.clip.config.vision_config.patch_size
        if freeze:
            self.clip.requires_grad_(False)
            self.clip.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.clip.eval()
        return self

    def forward(self, pixels, ids, mask):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            vi = self.clip.vision_model(pixel_values=pixels)
            te = self.clip.text_model(input_ids=ids, attention_mask=mask)
            # CLIP's final LayerNorm also normalizes the patch representation.
            patches = self.clip.vision_model.post_layernorm(vi.last_hidden_state[:, 1:])
            return (patches, self.clip.visual_projection(vi.pooler_output),
                    te.last_hidden_state, self.clip.text_projection(te.pooler_output),
                    (pixels.shape[-2] // self.patch_size, pixels.shape[-1] // self.patch_size))

class TinyEncoder(nn.Module):
    """Offline test fixture ONLY; randomly initialized, not a CLIP replacement."""
    def __init__(self):
        super().__init__()
        self.visual_dim = self.text_dim = self.global_dim = 32
        self.conv = nn.Conv2d(3, 32, 8, stride=8)
        self.embed = nn.Embedding(259, 32)

    def forward(self, pixels, ids, mask):
        x = self.conv(pixels)
        p = x.flatten(2).transpose(1, 2)
        t = self.embed(ids)
        g = (t * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return p, p.mean(1), t, g, x.shape[-2:]
