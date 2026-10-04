from PIL import Image
from torch.utils.data import Dataset
from .records import read_records, image_path, label_of
from .weak_supervision import make_weak_triple

class CosmosDataset(Dataset):
    def __init__(self, path, root, mode, pools=None, seed=42, same_image_weight=0.35):
        self.records = read_records(path)
        self.root, self.mode, self.pools, self.seed = root, mode, pools, seed
        self.same_image_weight = same_image_weight
        self.epoch = 0
        if mode != 'test':
            if any('articles' not in r or 'context_label' in r for r in self.records):
                raise ValueError('Weak training/validation require unlabeled articles records')
        else:
            for r in self.records:
                label_of(r)

    def __len__(self):
        return len(self.records) * (2 if self.mode != 'test' else 1)

    def __getitem__(self, index):
        ri = index // 2 if self.mode != 'test' else index
        r = self.records[ri]
        if self.mode == 'test':
            a, b, y, weight, origin = r['caption1'], r['caption2'], label_of(r), 1., 'human_test'
        else:
            a, b, y, weight, origin = make_weak_triple(
                r, self.pools, self.seed + self.epoch * 1000003 + ri * 2,
                bool(index % 2), self.same_image_weight)
        if not a.strip() or not b.strip():
            raise ValueError(f'Empty caption at {r["img_local_path"]}')
        with Image.open(image_path(self.root, r['img_local_path'])) as im:
            image = im.convert('RGB').copy()
        return dict(image=image, caption1=a, caption2=b, label=y, weight=weight,
                    group=ri, sample_id=r['img_local_path'], origin=origin)
