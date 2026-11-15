"""Full-resolution tiled inference; do not resize away fine vessels."""
import numpy as np
import torch
from PIL import Image


def choose_device(requested='auto'):
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def tile_starts(length, size, stride):
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length-size:
        starts.append(length-size)
    return starts


@torch.inference_mode()
def predict_probability(model, image, device, tile=384, overlap=64):
    if overlap < 0 or overlap >= tile:
        raise ValueError('overlap must be >= 0 and less than tile')
    array = np.asarray(image.convert('RGB'), dtype=np.float32)/255
    h, w = array.shape[:2]
    ph, pw = max(0, tile-h), max(0, tile-w)
    if ph or pw:
        array = np.pad(array, ((0,ph),(0,pw),(0,0)), mode='reflect')
    hh, ww = array.shape[:2]
    accumulated = np.zeros((hh, ww), dtype=np.float32)
    count = np.zeros((hh, ww), dtype=np.float32)
    for y in tile_starts(hh, tile, tile-overlap):
        for x in tile_starts(ww, tile, tile-overlap):
            crop = array[y:y+tile, x:x+tile].transpose(2,0,1).copy()
            tensor = torch.from_numpy(crop).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(tensor))[0,0].cpu().numpy()
            accumulated[y:y+tile, x:x+tile] += prob
            count[y:y+tile, x:x+tile] += 1
    return (accumulated / np.maximum(count, 1))[:h,:w]
