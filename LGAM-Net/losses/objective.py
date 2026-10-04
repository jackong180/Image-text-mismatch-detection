import torch
from torch.nn import functional as F

def multipositive_infonce(image, text, groups, temperature=0.07):
    scores = (image.float() @ text.float().T) / temperature
    positive = groups[:, None].eq(groups[None, :])
    def direction(s, p):
        return (torch.logsumexp(s, -1) - torch.logsumexp(s.masked_fill(~p, -torch.inf), -1)).mean()
    return 0.5 * (direction(scores, positive) + direction(scores.T, positive.T))

def objective(output, batch, config, augmented=None):
    raw = F.binary_cross_entropy_with_logits(output['logits'].float(), batch['labels'], reduction='none')
    cls = (raw * batch['weights']).sum() / batch['weights'].sum().clamp_min(1.)
    # caption1 always originates from this image, including corrupted triples.
    align = multipositive_infonce(output['image_embeddings'], output['text_embeddings'],
                                 batch['groups'], config.get('temperature', 0.07))
    consistency = output['logits'].sum() * 0.
    if augmented is not None:
        consistency = F.mse_loss(output['logits'].float().sigmoid(), augmented['logits'].float().sigmoid())
    total = cls + config.get('alignment', 0.1) * align + config.get('consistency', 0.05) * consistency
    return total, {'total': total.detach().item(), 'classification': cls.detach().item(),
                   'alignment': align.detach().item(), 'consistency': consistency.detach().item()}
