"""Export a vessel mask and overlay for a fundus image; RESEARCH-ONLY output."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from .evaluate import load_checkpoint
from .inference import choose_device,predict_probability


def predict(args):
    device=choose_device(args.device)
    model,ckpt=load_checkpoint(args.checkpoint,device)
    with Image.open(args.image) as im:
        rgb=im.convert('RGB')
    p=predict_probability(model,rgb,device,tile=int(ckpt['tile']),overlap=int(ckpt['overlap']))
    binary=p >= float(ckpt['threshold'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    Image.fromarray((binary*255).astype('uint8')).save(out/'vessels.png')
    overlay=np.asarray(rgb,dtype=np.float32).copy()
    overlay[binary]=.55*overlay[binary]+.45*np.array([255.,0.,0.])
    Image.fromarray(np.uint8(np.clip(overlay,0,255))).save(out/'overlay.png')
    print('Research-only output: ',out/'vessels.png',out/'overlay.png')


def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',default='outputs/prediction')
    p.add_argument('--device',default='auto')
    predict(p.parse_args())
if __name__=='__main__':main()
