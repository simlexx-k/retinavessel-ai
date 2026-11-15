"""Deterministic validation-only fundus image quality stress tests."""
from dataclasses import asdict, dataclass
import hashlib

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class DegradationCondition:
    name: str
    category: str
    severity: str
    value: float | None = None
    units: str | None = None


CONDITIONS = {
    condition.name: condition
    for condition in (
        DegradationCondition("clean", "none", "clean"),
        DegradationCondition("blur_mild", "blur", "mild", 1.0, "pixel radius"),
        DegradationCondition("blur_severe", "blur", "severe", 2.0, "pixel radius"),
        DegradationCondition("contrast_mild", "contrast", "mild", 0.75, "factor"),
        DegradationCondition("contrast_severe", "contrast", "severe", 0.55, "factor"),
        DegradationCondition(
            "illumination_mild", "illumination", "mild", 0.75, "brightness factor"
        ),
        DegradationCondition(
            "illumination_severe",
            "illumination",
            "severe",
            0.55,
            "brightness factor",
        ),
        DegradationCondition("noise_mild", "noise", "mild", 6.0, "RGB level sigma"),
        DegradationCondition(
            "noise_severe", "noise", "severe", 12.0, "RGB level sigma"
        ),
    )
}


def condition_manifest(names=None):
    """Return JSON-serializable definitions in the requested evaluation order."""
    selected = list(CONDITIONS) if names is None else list(names)
    unknown = set(selected) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown degradation conditions: {sorted(unknown)}")
    if len(selected) != len(set(selected)):
        raise ValueError("Degradation conditions must be unique")
    return [asdict(CONDITIONS[name]) for name in selected]


def stable_degradation_seed(base_seed, sample_id, condition):
    """Derive a platform-independent noise seed from the experiment identity."""
    payload = f"{int(base_seed)}\0{sample_id}\0{condition}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def apply_degradation(image, condition, seed=0):
    """Apply one fixed stress condition without changing image dimensions."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown degradation condition: {condition}")
    image = image.convert("RGB")
    specification = CONDITIONS[condition]
    if specification.category == "none":
        return image.copy()
    if specification.category == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=specification.value))
    if specification.category == "contrast":
        return ImageEnhance.Contrast(image).enhance(specification.value)
    if specification.category == "illumination":
        return ImageEnhance.Brightness(image).enhance(specification.value)
    if specification.category == "noise":
        array = np.asarray(image, dtype=np.float32)
        rng = np.random.default_rng(seed)
        noisy = np.clip(
            array + rng.normal(0.0, specification.value, array.shape), 0.0, 255.0
        ).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")
    raise AssertionError(f"Unhandled degradation category: {specification.category}")
