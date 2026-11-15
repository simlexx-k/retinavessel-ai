"""Train on image-level FIVES train/validation split; keep official test untouched."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    load_manifest,
    find_fives_root,
    FIVESPatchDataset,
    samples_from_frozen_split,
    read_pair,
)
from .inference import choose_device, predict_probability
from .metrics import approx_fov, compute_metrics
from .model import UNet, combined_loss, model_params


def validate(model, samples, device, tile, overlap, limit=None):
    model.eval()
    scores = []
    for sample in samples[:limit] if limit else samples:
        image, mask = read_pair(sample)
        p = predict_probability(model,image,device,tile=tile,overlap=overlap)
        fov = approx_fov(image)
        m = np.asarray(mask)>127
        scores.append(compute_metrics(p,m,fov)['dice'])
    return float(np.mean(scores)) if scores else 0.


def train(args):
    if args.train_limit is not None and args.train_limit <= 0:
        raise ValueError('--train-limit must be positive when supplied')
    if args.val_limit is not None and args.val_limit <= 0:
        raise ValueError('--val-limit must be positive when supplied')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    root=find_fives_root(args.data)
    official=load_manifest(root, 'train')
    split_path=Path(args.split_manifest)
    split_manifest=json.loads(split_path.read_text())
    train_samples,val_samples=samples_from_frozen_split(official,split_manifest)
    if not train_samples or not val_samples:
        raise ValueError('No samples in train/validation split')
    if args.train_limit:
        train_samples=train_samples[:args.train_limit]
    device=choose_device(args.device)
    model=UNet(base=args.base,lite=args.model=='lite_unet').to(device)
    print(f"device={device}, model={args.model}, params={model_params(model):,}")
    dataset=FIVESPatchDataset(train_samples,args.patch_size,args.patches_per_image,args.degradation)
    dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifest=dict(split_manifest)
    manifest['source_split_manifest']=str(split_path)
    manifest['training_run_seed']=args.seed
    manifest['training_run_train_count']=len(train_samples)
    manifest['training_run_validation_count']=min(len(val_samples),args.val_limit) if args.val_limit else len(val_samples)
    manifest['debug_limits']={'train_limit':args.train_limit,'validation_limit':args.val_limit}
    manifest['official_test_count_verified_at_training']=len(load_manifest(root,'test'))
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
    best=-1.0
    history=[]
    for epoch in range(1,args.epochs+1):
        model.train();total=0.;n=0
        for image,mask in tqdm(dataloader,desc=f"epoch {epoch}/{args.epochs}",leave=False):
            image=image.to(device);mask=mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(image)
            loss=combined_loss(logits,mask)
            loss.backward()
            optimizer.step()
            total+=float(loss.item());n+=1
        val_dice=validate(model,val_samples,device,args.patch_size,args.overlap,args.val_limit)
        record={'epoch':epoch,'training_loss':total/max(n,1),'validation_dice':val_dice}
        history.append(record)
        print(record)
        (out/'history.json').write_text(json.dumps(history,indent=2))
        if val_dice>best:
            best=val_dice
            torch.save({'model_state':model.state_dict(),'architecture':args.model,'base':args.base,
                        'tile':args.patch_size,'overlap':args.overlap,'threshold':0.5,
                        'seed':args.seed,'validation_dice':best,'epoch':epoch},out/'best.pt')
    return out/'best.pt'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True,help='Extracted original FIVES folder')
    p.add_argument('--model',choices=['unet','lite_unet'],default='unet')
    p.add_argument('--out',default='outputs/baseline')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--patch-size',type=int,default=384)
    p.add_argument('--patches-per-image',type=int,default=2)
    p.add_argument('--overlap',type=int,default=64)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--base',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--split-manifest',default='splits/fives_seed42.json',
                   help='Frozen audited train/validation manifest')
    p.add_argument('--val-limit',type=int,default=None,help='Debug only: limits validation images; omit for scientific experiments')
    p.add_argument('--train-limit',type=int,default=None,help='Debug only: limits fitting images; omit for scientific experiments')
    p.add_argument('--device',default='auto')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--degradation',action='store_true',help='Train with synthetic blur/brightness/contrast/noise; never modify masks')
    train(p.parse_args())
if __name__=='__main__':main()
