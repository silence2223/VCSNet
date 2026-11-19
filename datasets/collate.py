"""Expand training crops and pad image batches."""
from typing import List
import torch
from torch import Tensor

def collate_fn_crowd(batch, pad_block=128):
    """Flatten source-image crops into network samples."""
    batch_new = []
    for b in batch:
        imgs, points = b
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        for i in range(len(imgs)):

            batch_new.append((imgs[i, :, :, :], points[i]))
    batch = batch_new
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0], pad_block=pad_block)
    return tuple(batch)

def _max_by_axis_pad(the_list, block=128):
    """Find the maximum batch dimensions and round up to the padding block."""
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    if block > 1:
        for i in range(2):
            maxes[i + 1] = ((maxes[i + 1] - 1) // block + 1) * block
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor], pad_block: int=128):
    """Pad normalized tensors with zeros and return a [B,C,H,W] batch."""
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis_pad([list(img.shape) for img in tensor_list], block=pad_block)
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for img, pad_img in zip(tensor_list, tensor):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
    else:
        raise ValueError('not supported')
    return tensor
