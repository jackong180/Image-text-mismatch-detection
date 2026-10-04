import numpy as np
import torch
from PIL import Image

class BatchCollator:
    def __init__(self, model_config):
        self.tiny = model_config.get('encoder') == 'tiny'
        self.max_length = model_config.get('max_length', 77)
        self.processor = None
        if not self.tiny:
            from transformers import CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(model_config['clip_name'],
                local_files_only=model_config.get('local_files_only', False))
            if not 3 <= self.max_length <= 77:
                raise ValueError('CLIP max_length must be in [3,77]')

    def __call__(self, rows):
        texts = [r['caption1'] for r in rows] + [r['caption2'] for r in rows]
        if self.tiny:
            ids = torch.zeros(len(texts), 32, dtype=torch.long)
            mask = torch.zeros_like(ids)
            special = torch.ones_like(ids, dtype=torch.bool)
            for i, text in enumerate(texts):
                tokens = [257] + [x + 1 for x in text.encode('utf-8')[:30]] + [258]
                ids[i, :len(tokens)] = torch.tensor(tokens)
                mask[i, :len(tokens)] = 1
                special[i, 1:len(tokens)-1] = False
            pixels = torch.stack([torch.from_numpy(np.asarray(r['image'].resize((32, 32))).copy()).permute(2,0,1).float()/255 for r in rows])
        else:
            # Letterbox retains edge evidence; disable CLIP's default center crop.
            images = []
            for r in rows:
                im = r['image'].copy()
                im.thumbnail((224, 224))
                canvas = Image.new('RGB', (224,224), (123,117,104))
                canvas.paste(im, ((224-im.width)//2, (224-im.height)//2))
                images.append(canvas)
            pixels = self.processor.image_processor(images=images, do_resize=False,
                        do_center_crop=False, return_tensors='pt')['pixel_values']
            tokens = self.processor.tokenizer(texts, padding='max_length', truncation=True,
                        max_length=self.max_length, return_special_tokens_mask=True, return_tensors='pt')
            ids, mask = tokens['input_ids'], tokens['attention_mask']
            special = tokens['special_tokens_mask'].bool()
        local = mask.bool() & ~special
        if not local.any(-1).all():
            raise ValueError('Caption has no usable local tokens')
        n = len(rows)
        return dict(pixels=pixels, ids1=ids[:n], ids2=ids[n:], mask1=mask[:n], mask2=mask[n:],
                    local_mask1=local[:n], local_mask2=local[n:],
                    labels=torch.tensor([r['label'] for r in rows], dtype=torch.float32),
                    weights=torch.tensor([r['weight'] for r in rows], dtype=torch.float32),
                    groups=torch.tensor([r['group'] for r in rows]),
                    sample_ids=[r['sample_id'] for r in rows], origins=[r['origin'] for r in rows])
