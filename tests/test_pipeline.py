import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from retinavessel.data import (
    Sample,
    load_manifest,
    find_fives_root,
    stratified_train_val,
    quality_stratified_train_val,
    samples_from_frozen_split,
    FIVESPatchDataset,
)
from retinavessel.quality import QualityScore
from retinavessel.model import UNet,combined_loss
from retinavessel.inference import predict_probability
from retinavessel.metrics import compute_metrics,approx_fov
from retinavessel.audit import audit


def make_fixture(tmp_path):
    root=tmp_path/'FIVES'
    for split in ('train','test'):
        (root/split/'Original').mkdir(parents=True)
        (root/split/'Ground Truth').mkdir(parents=True)
        for i,cond in enumerate('ADGN',1):
            rgb=np.zeros((64,64,3),dtype='uint8')
            rgb[3:61,3:61,:]=120
            m=np.zeros((64,64),dtype='uint8')
            m[20:44,31:33]=255
            Image.fromarray(rgb).save(root/split/'Original'/f'{i}_{cond}.png')
            Image.fromarray(m).save(root/split/'Ground Truth'/f'{i}_{cond}.png')
    return root


def test_data_and_audit(tmp_path):
    root=make_fixture(tmp_path)
    samples=load_manifest(root,'train')
    assert len(samples)==4
    train,val=stratified_train_val(samples)
    assert len(train)==0 and len(val)==4 # Tiny fixture, 1 validation image per class
    ds=FIVESPatchDataset(samples,patch_size=64)
    a,b=ds[0]
    assert a.shape==(3,64,64) and b.shape==(1,64,64)
    result=audit(root,tmp_path/'audit.json')
    assert result['total_paired_images']==8
    assert len(result['identical_pixels_cross_split'])==1  # Four identical synthetic originals
    assert (tmp_path/'audit.json').exists()


def test_loader_accepts_official_ground_truth_casing(tmp_path):
    root=make_fixture(tmp_path)
    (root/'train'/'Ground Truth').rename(root/'train'/'Ground truth')
    assert find_fives_root(tmp_path)==root
    assert len(load_manifest(root,'train'))==4


def test_quality_stratified_frozen_split_is_exact_and_deterministic(tmp_path):
    samples=[]
    quality={}
    for condition in 'ADGN':
        for number in range(1,21):
            sample_id=f'{number}_{condition}'
            samples.append(Sample(tmp_path/f'{sample_id}.png',tmp_path/f'{sample_id}_mask.png','train',condition))
            quality[sample_id]=QualityScore(0,1,1) if number<=5 else QualityScore(1,1,1)
    train_a,val_a=quality_stratified_train_val(
        samples,quality,seed=42,val_fraction=.15,excluded_ids={'20_N'}
    )
    train_b,val_b=quality_stratified_train_val(
        samples,quality,seed=42,val_fraction=.15,excluded_ids={'20_N'}
    )
    assert len(train_a)==67 and len(val_a)==12
    assert [sample.id for sample in train_a]==[sample.id for sample in train_b]
    assert [sample.id for sample in val_a]==[sample.id for sample in val_b]
    assert '20_N' not in {sample.id for sample in train_a+val_a}
    manifest={
        'train':[sample.id for sample in train_a],
        'validation':[sample.id for sample in val_a],
        'excluded':{'20_N':'fixture exclusion'},
    }
    frozen_train,frozen_val=samples_from_frozen_split(samples,manifest)
    assert [sample.id for sample in frozen_train]==manifest['train']
    assert [sample.id for sample in frozen_val]==manifest['validation']


def test_train_predict_and_metrics():
    model=UNet(base=4)
    x=torch.rand((1,3,64,64));y=torch.zeros((1,1,64,64))
    logits=model(x)
    assert logits.shape==y.shape
    loss=combined_loss(logits,y)
    loss.backward();assert torch.isfinite(loss)
    rgb=np.full((64,64,3),100,dtype='uint8')
    p=predict_probability(model.eval(),Image.fromarray(rgb),torch.device('cpu'),tile=64,overlap=16)
    assert p.shape==(64,64) and np.isfinite(p).all()
    result=compute_metrics(np.ones((64,64)),np.ones((64,64),dtype=bool),np.ones((64,64),dtype=bool),include_topology=True)
    assert result['dice']==1 and result['cldice']==1
    assert approx_fov(Image.fromarray(rgb)).any()
