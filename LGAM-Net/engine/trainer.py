import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.dataset import CosmosDataset
from data.collate import BatchCollator
from data.records import read_records
from data.weak_supervision import build_entity_pool, make_weak_triple
from losses.objective import objective
from models.lgam_net import LGAMNet
from utils.runtime import seed_everything, worker_seed, to_device, get_device, write_json, save_checkpoint


def run_epoch(model, loader, device, config, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = {k: 0. for k in ('total','classification','alignment','consistency')}
    count = 0
    accumulation = config['train']['accumulation']
    amp = config['train'].get('amp', True) and device.type == 'cuda'
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, raw in enumerate(loader):
        batch = to_device(raw, device)
        # Last partial accumulation group receives its actual group size.
        group_start = (step // accumulation) * accumulation
        group_size = min(accumulation, len(loader) - group_start)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp):
            result = model(batch)
            augmented = None
            if training and config['loss'].get('consistency', 0) > 0:
                aug_batch = dict(batch)
                # Mild brightness shift in normalized tensor space; no spatial crop.
                shift = (torch.rand(batch['pixels'].shape[0],1,1,1,device=device) - .5) * .08
                aug_batch['pixels'] = batch['pixels'] + shift
                augmented = model(aug_batch)
            loss, pieces = objective(result, batch, config['loss'], augmented)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at batch {step}')
        if training:
            scaler.scale(loss / group_size).backward()
            if (step+1) % accumulation == 0 or step+1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train'].get('grad_clip',1.))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        n = len(batch['labels'])
        count += n
        for k in totals:
            totals[k] += pieces[k] * n
        if training and (step+1) % config['train'].get('log_every',50) == 0:
            print(f'  batch {step+1}/{len(loader)} loss={pieces["total"]:.4f}', flush=True)
    if not count:
        raise ValueError('Empty loader')
    return {k: v/count for k,v in totals.items()}


def train(config, resume=None):
    seed_everything(config['seed'])
    device = get_device(config.get('device','auto'))
    d, t = config['data'], config['train']
    train_records, val_records = read_records(d['train_json']), read_records(d['val_json'])
    train_paths = {r['img_local_path'] for r in train_records}
    if train_paths & {r['img_local_path'] for r in val_records}:
        raise ValueError('Train/validation image paths overlap')
    pools = build_entity_pool(train_records)  # Never uses test or validation entities.
    if not any(len(v)>1 for v in pools.values()):
        raise ValueError('No usable training entity pool for synthetic negatives')
    train_set = CosmosDataset(d['train_json'], d['root'], 'train', pools, config['seed'],d.get('same_image_weight',.35))
    val_set = CosmosDataset(d['val_json'], d['root'], 'val', pools, config['seed']+7919,d.get('same_image_weight',.35))
    collate = BatchCollator(config['model'])
    generator = torch.Generator().manual_seed(config['seed'])
    args = dict(batch_size=t['batch_size'], num_workers=t.get('workers',0), collate_fn=collate,
                pin_memory=device.type=='cuda', worker_init_fn=worker_seed, persistent_workers=False)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **args)
    val_loader = DataLoader(val_set, shuffle=False, **args)
    model = LGAMNet(config['model']).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=t['lr'],weight_decay=t.get('weight_decay',.01))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=t['epochs'])
    scaler = torch.amp.GradScaler('cuda',enabled=t.get('amp',True) and device.type=='cuda')
    start, best = 0, math.inf
    out = Path(t['output']); out.mkdir(parents=True,exist_ok=True)
    if not resume and (out/'last.pt').exists():
        raise FileExistsError('Output contains checkpoint: use --resume or a new output folder')
    if resume:
        checkpoint = torch.load(resume,map_location='cpu',weights_only=False)
        if checkpoint['config']['model'] != config['model'] or checkpoint['config']['data'] != config['data']:
            raise ValueError('Resume model/data configuration differs')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start, best = checkpoint['epoch']+1, checkpoint['best_proxy_loss']
        torch.set_rng_state(checkpoint['torch_rng'])
        random.setstate(checkpoint['python_rng']); np.random.set_state(checkpoint['numpy_rng'])
        generator.set_state(checkpoint['loader_rng'])
        if device.type=='cuda' and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    write_json(out/'config.json',config)
    write_json(out/'entity_pool.json',pools)
    # Human-readable audit of the initial synthetic supervision, without images.
    audit = []
    for i, record in enumerate(train_records[:100]):
        for negative in (False, True):
            a,b,y,w,origin = make_weak_triple(record,pools,config['seed']+i*2,negative,d.get('same_image_weight',.35))
            audit.append(dict(sample_id=record['img_local_path'],caption1=a,caption2=b,
                              pseudo_label=y,weight=w,origin=origin))
    write_json(out/'weak_supervision_audit.json',audit)
    print(f'Device={device}; trainable parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}',flush=True)
    for epoch in range(start,t['epochs']):
        train_set.epoch = epoch
        tr = run_epoch(model,train_loader,device,config,optimizer,scaler)
        va = run_epoch(model,val_loader,device,config)
        scheduler.step()
        improved = va['total'] < best
        best = min(best,va['total'])
        report = {'epoch':epoch+1,'train':tr,'validation_proxy':va,'lr':optimizer.param_groups[0]['lr']}
        print(json.dumps(report),flush=True)
        with (out/'history.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(report)+'\n')
        payload = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                       scaler=scaler.state_dict(),config=config,epoch=epoch,best_proxy_loss=best,
                       torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                       loader_rng=generator.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        save_checkpoint(out/'last.pt',payload)
        if improved:
            save_checkpoint(out/'best.pt',payload)
    return out/'best.pt'
