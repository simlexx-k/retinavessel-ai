"""Evaluate model on untouched official FIVES test split, full native resolution."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .data import load_manifest, read_pair
from .inference import choose_device,predict_probability
from .metrics import compute_metrics,approx_fov
from .model import UNet,model_params


def load_checkpoint(file, device):
    ckpt=torch.load(file,map_location='cpu',weights_only=True)
    model=UNet(base=int(ckpt['base']),lite=ckpt['architecture']=='lite_unet')
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    return model,ckpt


def evaluate(args):
    samples=load_manifest(args.data,'test')
    device=choose_device(args.device)
    model,ckpt=load_checkpoint(args.checkpoint,device)
    tile=int(ckpt['tile']);overlap=int(ckpt['overlap']);threshold=float(ckpt['threshold'])
    records=[]
    for i,s in enumerate(samples,1):
        image,mask=read_pair(s)
        start=time.perf_counter()
        probs=predict_probability(model,image,device,tile,overlap)
        elapsed=time.perf_counter()-start
        result=compute_metrics(probs,np.asarray(mask)>127,approx_fov(image),threshold,include_topology=args.topology)
        result.update({'image_id':s.id,'condition':s.condition,'seconds':elapsed})
        records.append(result)
        if i % 10 == 0:print(f'Processed {i}/{len(samples)}')
    p=Path(args.out);p.mkdir(parents=True,exist_ok=True)
    with (p/'test_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0].keys()))
        writer.writeheader();writer.writerows(records)
    sums={k:sum(r[k] for r in records) for k in ('tp','fp','tn','fn')}
    tp,fp,tn,fn=(sums[k] for k in ('tp','fp','tn','fn'))
    def safe(a,b):return round(float(a/b),6) if b else 0.
    report={'model':ckpt['architecture'],'parameters':model_params(model),'dataset':'FIVES official test',
            'image_count':len(records),'decision_threshold':threshold,'pixel_micro_dice':safe(2*tp,2*tp+fp+fn),
            'pixel_micro_iou':safe(tp,tp+fp+fn),'pixel_sensitivity':safe(tp,tp+fn),
            'pixel_specificity':safe(tn,tn+fp),'per_image_mean_dice':round(float(np.mean([r['dice'] for r in records])),6),
            'per_image_median_dice':round(float(np.median([r['dice'] for r in records])),6),
            'mean_seconds_per_image':round(float(np.mean([r['seconds'] for r in records])),3),
            'by_condition':{k:{'count':sum(r['condition']==k for r in records),
                               'mean_dice':round(float(np.mean([r['dice'] for r in records if r['condition']==k])),6)} for k in 'ADGN'},
            'caveats':['FOV is image-derived; inspect against provided or manually defined reference FOV masks.',
                       'FIVES patient identifiers are not supplied: no claim of patient-independent train/test split.',
                       'Audit anomalies before using results; do not tune on official test set.',
                       'Research-only segmentation; no clinical performance claims.']}
    if args.topology: report['per_image_mean_cldice']=round(float(np.mean([r['cldice'] for r in records])),6)
    (p/'test_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',default='outputs/evaluation')
    p.add_argument('--device',default='auto')
    p.add_argument('--topology',action='store_true',help='Compute clDice (slower)')
    evaluate(p.parse_args())
if __name__=='__main__':main()
