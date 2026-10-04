import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data.dataset import CosmosDataset
from data.collate import BatchCollator
from models.lgam_net import LGAMNet
from utils.runtime import get_device, to_device, write_json
from utils.metrics import classification_metrics


def evaluate(checkpoint_path, test_json, data_root, output, device_name='auto', batch_size=16, workers=0, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0,1]')
    device = get_device(device_name)
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    config = checkpoint['config']
    model = LGAMNet(config['model']).to(device)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    dataset = CosmosDataset(test_json,data_root,'test')
    loader = DataLoader(dataset,batch_size=batch_size,num_workers=workers,shuffle=False,
                        collate_fn=BatchCollator(config['model']))
    rows, labels, probabilities = [], [], []
    with torch.inference_mode():
        for raw in loader:
            b = to_device(raw,device)
            probs = model(b)['logits'].sigmoid().cpu().tolist()
            ys = raw['labels'].int().tolist()
            labels.extend(ys); probabilities.extend(probs)
            for sample,y,p in zip(raw['sample_ids'],ys,probs):
                pred = int(p>=threshold)
                case = ('TP' if pred else 'FN') if y else ('FP' if pred else 'TN')
                rows.append(dict(sample_id=sample,label=y,probability=p,prediction=pred,case=case))
    metrics = classification_metrics(labels,probabilities,threshold)
    metrics.update(checkpoint=str(checkpoint_path),test_json=str(test_json),encoder=config['model'].get('encoder','clip'))
    out = Path(output); out.mkdir(parents=True,exist_ok=True)
    write_json(out/'metrics.json',metrics)
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return metrics
