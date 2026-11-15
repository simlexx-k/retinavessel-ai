"""Train on image-level FIVES train/validation split; keep official test untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import random
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


def _training_config(args, split_path):
    """Return the settings that must match for an exact training resume."""
    return {
        'model': args.model,
        'base': args.base,
        'batch_size': args.batch_size,
        'patch_size': args.patch_size,
        'patches_per_image': args.patches_per_image,
        'overlap': args.overlap,
        'learning_rate': args.lr,
        'seed': args.seed,
        'degradation': args.degradation,
        'patience': args.patience,
        'train_limit': args.train_limit,
        'validation_limit': args.val_limit,
        'split_manifest_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }


def _rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def _atomic_torch_save(payload, path):
    """Avoid leaving a partially written checkpoint after runtime interruption."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_resume(path, model, optimizer, expected_config):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    required = {
        'model_state', 'optimizer_state', 'epoch', 'history',
        'best_validation_dice', 'epochs_without_improvement',
        'training_config', 'rng_state',
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            f"Resume checkpoint is missing {sorted(missing)}. "
            "Legacy best.pt files cannot exactly resume training; start a new run."
        )
    if checkpoint['training_config'] != expected_config:
        differing = sorted(
            key for key in set(checkpoint['training_config']) | set(expected_config)
            if checkpoint['training_config'].get(key) != expected_config.get(key)
        )
        raise ValueError(f'Resume configuration mismatch: {differing}')
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    _restore_rng_state(checkpoint['rng_state'])
    return checkpoint


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
    if args.patience is not None and args.patience <= 0:
        raise ValueError('--patience must be positive when supplied')
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
    training_config=_training_config(args,split_path)
    manifest=dict(split_manifest)
    manifest['source_split_manifest']=str(split_path)
    manifest['training_run_seed']=args.seed
    manifest['training_run_train_count']=len(train_samples)
    manifest['training_run_validation_count']=min(len(val_samples),args.val_limit) if args.val_limit else len(val_samples)
    manifest['debug_limits']={'train_limit':args.train_limit,'validation_limit':args.val_limit}
    manifest['official_test_count_verified_at_training']=len(load_manifest(root,'test'))
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
    best=-1.0
    epochs_without_improvement=0
    history=[]
    start_epoch=1
    if args.resume:
        resume_path=Path(args.resume)
        if resume_path.resolve().parent!=out.resolve():
            raise ValueError('--resume must point to last.pt inside --out')
        if not (out/'best.pt').is_file():
            raise ValueError('Resumed output directory must also contain best.pt')
        checkpoint=_load_resume(
            resume_path,model,optimizer,training_config
        )
        start_epoch=int(checkpoint['epoch'])+1
        best=float(checkpoint['best_validation_dice'])
        epochs_without_improvement=int(checkpoint['epochs_without_improvement'])
        history=list(checkpoint['history'])
        if start_epoch>args.epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {start_epoch-1}; "
                f"--epochs={args.epochs} must be larger"
            )
        print(f"resuming from epoch {start_epoch-1}; target epoch={args.epochs}")
    for epoch in range(start_epoch,args.epochs+1):
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
        improved=val_dice>best
        if improved:
            best=val_dice
            epochs_without_improvement=0
            _atomic_torch_save(
                {'model_state':model.state_dict(),'architecture':args.model,'base':args.base,
                 'tile':args.patch_size,'overlap':args.overlap,'threshold':0.5,
                 'seed':args.seed,'validation_dice':best,'epoch':epoch},
                out/'best.pt'
            )
        else:
            epochs_without_improvement+=1
        _atomic_torch_save(
            {
                'model_state':model.state_dict(),
                'optimizer_state':optimizer.state_dict(),
                'epoch':epoch,
                'history':history,
                'best_validation_dice':best,
                'epochs_without_improvement':epochs_without_improvement,
                'training_config':training_config,
                'rng_state':_rng_state(),
            },
            out/'last.pt'
        )
        if args.patience is not None and epochs_without_improvement>=args.patience:
            print(
                f"early stopping at epoch {epoch}; best validation Dice={best:.6f}"
            )
            break
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
    p.add_argument('--patience',type=int,default=None,
                   help='Stop after this many epochs without validation Dice improvement')
    p.add_argument('--resume',default=None,
                   help='Exact-resume from last.pt; --epochs is the new total target')
    train(p.parse_args())
if __name__=='__main__':main()
