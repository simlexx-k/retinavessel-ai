"""Metadata/data-integrity gate for the original FIVES dataset.

Flags anomalies, never silently removes records or claims labels are correct.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .data import find_fives_root, load_manifest
from .metrics import approx_fov


def audit(root, out):
    root = find_fives_root(root)
    samples = load_manifest(root, 'train') + load_manifest(root,'test')
    results = []
    image_hash_groups = defaultdict(list)
    for s in samples:
        with Image.open(s.image) as im:
            image = im.convert('RGB')
        with Image.open(s.mask) as ma:
            mask = ma.convert('L')
        mismatch = image.size != mask.size
        if mismatch:
            m = None
            foreground_fraction = None
        else:
            m = (np.asarray(mask) > 127)
            try:
                fov = approx_fov(image)
                foreground_fraction = round(float((m & fov).sum()/max(fov.sum(),1)),6)
            except ValueError:
                foreground_fraction = None
        pixel_hash = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        mask_hash = hashlib.sha256(np.asarray(mask).tobytes()).hexdigest()
        image_hash_groups[pixel_hash].append((s.split, s.id, mask_hash))
        results.append({'split': s.split, 'id': s.id,'condition':s.condition,
                        'size': image.size,'mask_size':mask.size, 'shape_mismatch':mismatch,
                        'vessel_fraction_inside_approx_fov':foreground_fraction,
                        'all_background_mask': bool(not m.any()) if m is not None else None,
                        'all_foreground_mask': bool(m.all()) if m is not None else None,
                        'pixel_hash':pixel_hash, 'mask_hash':mask_hash})
    repeated = [grp for grp in image_hash_groups.values() if len(grp)>1]
    cross_split = [grp for grp in repeated if len(set(s for s,_,_ in grp))>1]
    conflicting = [grp for grp in repeated if len(set(m for _,_,m in grp))>1]
    stats = {'total_paired_images':len(results),
             'train': sum(x['split']=='train' for x in results),
             'test': sum(x['split']=='test' for x in results),
             'missing_or_mismatched_sizes':sum(x['shape_mismatch'] for x in results),
             'all_background_masks':sum(x['all_background_mask'] is True for x in results),
             'all_foreground_masks':sum(x['all_foreground_mask'] is True for x in results),
             'identical_pixel_groups':repeated, 'identical_pixels_cross_split':cross_split,
             'identical_pixels_conflicting_masks':conflicting,
             'suspicious_low_vessel_fraction':[(x['split'],x['id']) for x in results if x['vessel_fraction_inside_approx_fov'] is not None and x['vessel_fraction_inside_approx_fov'] < 0.005],
             'suspicious_high_vessel_fraction':[(x['split'],x['id']) for x in results if x['vessel_fraction_inside_approx_fov'] is not None and x['vessel_fraction_inside_approx_fov'] > 0.3],
             'warning':'Review all flagged entries manually; thresholds are heuristic, not grounds for automatic deletion.'}
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'summary':stats,'samples':results},indent=2))
    print(json.dumps(stats,indent=2))
    return stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='outputs/fives_audit.json')
    a=p.parse_args();audit(a.data,a.out)
if __name__=='__main__':main()
