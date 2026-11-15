"""FIVES official-directory loader: train/Original and train/Ground truth, etc."""
from dataclasses import dataclass
from collections import defaultdict
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    split: str
    condition: str
    @property
    def id(self):
        return self.image.stem


def find_fives_root(path):
    path = Path(path).expanduser().resolve()

    def has_train_layout(candidate):
        train_dir = candidate / "train"
        return (_find_directory(train_dir, "Original") is not None
                and _find_directory(train_dir, "Ground Truth") is not None)

    if has_train_layout(path):
        return path
    hits = [p.parent for p in path.rglob("train") if p.is_dir()
            and has_train_layout(p.parent)]
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(
        f"Expected FIVES train/Original and train/Ground truth under {path}; "
        f"found {len(hits)} roots"
    )


def _find_directory(parent, name):
    """Return one child directory by case-insensitive name."""
    if not parent.is_dir():
        return None
    matches = [p for p in parent.iterdir()
               if p.is_dir() and p.name.casefold() == name.casefold()]
    return matches[0] if len(matches) == 1 else None


def load_manifest(root, split):
    root = find_fives_root(root)
    split_dir = root / split
    image_dir = _find_directory(split_dir, "Original")
    mask_dir = _find_directory(split_dir, "Ground Truth")
    if image_dir is None or mask_dir is None:
        raise FileNotFoundError(f"Missing image/mask folder in {root / split}")
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")])
    mask_lookup = {p.stem.lower(): p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")}
    out = []
    for image in images:
        mask = mask_lookup.get(image.stem.lower())
        if mask is None:
            raise ValueError(f"Missing ground-truth mask for {image}")
        condition = image.stem.split("_")[-1].upper()
        if condition not in ("A", "D", "G", "N"):
            raise ValueError(f"Cannot read disease grouping from FIVES filename: {image.name}")
        out.append(Sample(image, mask, split, condition))
    if not out:
        raise ValueError(f"No paired image files found in {image_dir}")
    return out


def stratified_train_val(samples, seed=42, val_fraction=0.15):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    rng = random.Random(seed)
    train, val = [], []
    for condition in "ADGN":
        members = [s for s in samples if s.condition == condition]
        if not members:
            continue
        rng.shuffle(members)
        nval = max(1, int(round(len(members) * val_fraction)))
        val.extend(members[:nval])
        train.extend(members[nval:])
    return sorted(train, key=lambda x:x.id), sorted(val, key=lambda x:x.id)


def quality_stratified_train_val(samples, quality_scores, seed=42,
                                 val_fraction=0.15, excluded_ids=()):
    """Make an exact-size split stratified by condition and quality class.

    Quality class is deliberately coarse: all three official quality fields are
    good versus at least one field being poor. This avoids tiny strata while
    preserving low-quality examples in both training and validation.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    excluded_ids = set(excluded_ids)
    unknown_exclusions = excluded_ids - {sample.id for sample in samples}
    if unknown_exclusions:
        raise ValueError(f"Unknown excluded sample IDs: {sorted(unknown_exclusions)}")
    eligible = [sample for sample in samples if sample.id not in excluded_ids]
    missing_quality = [sample.id for sample in eligible
                       if sample.id not in quality_scores]
    if missing_quality:
        raise ValueError(f"Missing quality scores: {missing_quality[:10]}")

    strata = defaultdict(list)
    for sample in eligible:
        quality_class = "all_good" if quality_scores[sample.id].all_good else "degraded"
        strata[(sample.condition, quality_class)].append(sample)
    if any(len(members) < 2 for members in strata.values()):
        small = {str(key): len(value) for key, value in strata.items()
                 if len(value) < 2}
        raise ValueError(f"Cannot split strata with fewer than two samples: {small}")

    target_validation = int(round(len(eligible) * val_fraction))
    if target_validation < len(strata) or len(eligible) - target_validation < len(strata):
        raise ValueError(
            "Requested split is too small to place every disease/quality stratum "
            "in both training and validation"
        )
    raw = {key: len(members) * val_fraction for key, members in strata.items()}
    quotas = {
        key: max(1, min(len(strata[key]) - 1, math.floor(value)))
        for key, value in raw.items()
    }
    while sum(quotas.values()) < target_validation:
        candidates = [key for key in strata if quotas[key] < len(strata[key]) - 1]
        key = max(candidates, key=lambda item: (raw[item] - quotas[item], item))
        quotas[key] += 1
    while sum(quotas.values()) > target_validation:
        candidates = [key for key in strata if quotas[key] > 1]
        key = max(candidates, key=lambda item: (quotas[item] - raw[item], item))
        quotas[key] -= 1

    rng = random.Random(seed)
    train, validation = [], []
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda sample: sample.id)
        rng.shuffle(members)
        validation.extend(members[:quotas[key]])
        train.extend(members[quotas[key]:])
    return (sorted(train, key=lambda sample: sample.id),
            sorted(validation, key=lambda sample: sample.id))


def samples_from_frozen_split(samples, manifest):
    """Resolve and validate train/validation IDs from a frozen manifest."""
    lookup = {sample.id: sample for sample in samples}
    train_ids = manifest.get("train", [])
    validation_ids = manifest.get("validation", [])
    if len(train_ids) != len(set(train_ids)) or len(validation_ids) != len(set(validation_ids)):
        raise ValueError("Frozen split contains duplicate IDs")
    overlap = set(train_ids) & set(validation_ids)
    if overlap:
        raise ValueError(f"Frozen train/validation overlap: {sorted(overlap)}")
    unknown = (set(train_ids) | set(validation_ids)) - set(lookup)
    if unknown:
        raise ValueError(f"Frozen split contains unknown IDs: {sorted(unknown)}")
    excluded = set(manifest.get("excluded", {}))
    if excluded & (set(train_ids) | set(validation_ids)):
        raise ValueError("Frozen split includes an explicitly excluded sample")
    return ([lookup[sample_id] for sample_id in train_ids],
            [lookup[sample_id] for sample_id in validation_ids])


def read_pair(sample):
    with Image.open(sample.image) as im:
        rgb = im.convert("RGB")
    with Image.open(sample.mask) as m:
        mask = m.convert("L")
    if rgb.size != mask.size:
        raise ValueError(f"Image/mask dimensions mismatch: {sample.id}: {rgb.size}, {mask.size}")
    return rgb, mask


class FIVESPatchDataset(Dataset):
    def __init__(self, samples, patch_size=384, patches_per_image=2, degradation=False):
        if patch_size < 64 or patch_size % 16:
            raise ValueError("patch_size must be >=64 and divisible by 16")
        self.samples = list(samples)
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.degradation = degradation

    def __len__(self):
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, index):
        s = self.samples[index % len(self.samples)]
        image, mask = read_pair(s)
        w, h = image.size
        p = self.patch_size
        if min(w, h) < p:
            raise ValueError(f"Image smaller than crop size: {s.id}")
        x = random.randint(0, w - p)
        y = random.randint(0, h - p)
        image = image.crop((x, y, x+p, y+p))
        mask = mask.crop((x, y, x+p, y+p))
        if random.random() < 0.5:
            image, mask = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image, mask = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        k = random.randrange(4)
        if k:
            image, mask = image.rotate(90*k), mask.rotate(90*k)
        if self.degradation and random.random() < 0.7:
            image = random_degrade(image)
        a = np.asarray(image, dtype=np.float32).copy()/255.0
        m = (np.asarray(mask) > 127).astype(np.float32).copy()
        return torch.from_numpy(a.transpose(2,0,1)), torch.from_numpy(m[None])


def random_degrade(image):
    """Mild *training-only* synthetic degradations; ground-truth mask unchanged."""
    operation = random.choice(("blur", "contrast", "brightness", "noise"))
    if operation == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.8)))
    if operation == "contrast":
        return ImageEnhance.Contrast(image).enhance(random.uniform(0.55, 0.9))
    if operation == "brightness":
        return ImageEnhance.Brightness(image).enhance(random.uniform(0.65, 1.25))
    array = np.asarray(image, dtype=np.float32)
    array = np.clip(array + np.random.normal(0, random.uniform(3, 12), array.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(array)
