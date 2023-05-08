import io
import psutil
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from torch.linalg import norm
import PIL.Image
import gym.spaces
from mpi4py import MPI


# Note! This is l2 square, not l2
def l2(a, b):
    return torch.pow(torch.abs(a - b), 2).sum(dim=1)


# required when we load optimizer from a checkpoint
def optimizer_cuda(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def get_ckpt_path(base_dir, ckpt_num):
    base_dir = Path(base_dir)
    if ckpt_num is None:
        if base_dir.suffix == ".pt":
            max_step = str(base_dir).rsplit("_", 1)[-1].split(".")[0]
            return base_dir, max_step
        return get_recent_ckpt_path(base_dir)
    files = base_dir.glob("*.pt")
    for f in files:
        if f"ckpt_{ckpt_num:011d}.pt" in str(f):
            return f, ckpt_num
    raise Exception(f"Did not find ckpt_{ckpt_num}.pt")


def get_recent_ckpt_path(base_dir):
    files = list(Path(base_dir).glob("*.pt"))
    files.sort()
    if len(files) == 0:
        return None, None
    max_step = max([str(f).rsplit("_", 1)[-1].split(".")[0] for f in files])
    paths = [f for f in files if max_step in str(f)]
    if len(paths) == 1:
        return paths[0], int(max_step)
    else:
        raise Exception(f"Multiple most recent ckpts {paths}")


def image_grid(image, n=4):
    return vutils.make_grid(image[:n], nrow=n).cpu().detach().numpy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def slice_tensor(input, indices):
    ret = {}
    for k, v in input.items():
        ret[k] = v[indices]
    return ret


def average_gradients(model):
    size = float(dist.get_world_size())
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data /= size


def ensure_shared_grads(model, shared_model):
    """for A3C"""
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def compute_gradient_norm(model):
    ps = model.parameters()
    return norm(torch.stack([norm(p.grad.detach()) for p in ps if p.grad is not None]))


def compute_weight_norm(model):
    ps = model.parameters()
    return norm(torch.stack([norm(p.data.detach()) for p in ps if p.data is not None]))


def compute_weight_sum(model):
    ps = model.parameters()
    return torch.stack([p.data.detach().sum() for p in ps if p.data is not None]).sum()


def sync_network(network, device):
    """Sync networks across the different cores."""
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 1:
        return
    flat_params, params_shape = _get_flat_params(network)
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params(network, params_shape, flat_params, device)


def _get_flat_params(network):
    """Get the flat params from the network."""
    param_shape = {}
    flat_params = None
    for key_name, value in network.named_parameters():
        param_shape[key_name] = value.cpu().detach().numpy().shape
        if flat_params is None:
            flat_params = value.cpu().detach().numpy().flatten()
        else:
            flat_params = np.append(flat_params, value.cpu().detach().numpy().flatten())
    return flat_params, param_shape


def _set_flat_params(network, params_shape, params, device):
    """Sets the params from the network."""
    pointer = 0
    for key_name, values in network.named_parameters():
        # get the length of the parameters
        len_param = np.prod(params_shape[key_name])
        copy_params = params[pointer : pointer + len_param].reshape(
            params_shape[key_name]
        )
        copy_params = torch.tensor(copy_params).to(device)
        # copy the params
        values.data.copy_(copy_params.data)
        # update the pointer
        pointer += len_param


def sync_grad(network, device):
    """Sync gradients across the different cores."""
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 1:
        return
    flat_grads, grads_shape = _get_flat_grads(network)
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    _set_flat_grads(network, grads_shape, global_grads, device)


def _set_flat_grads(network, grads_shape, flat_grads, device):
    pointer = 0
    for key_name, value in network.named_parameters():
        if key_name in grads_shape:
            len_grads = np.prod(grads_shape[key_name])
            copy_grads = flat_grads[pointer : pointer + len_grads].reshape(
                grads_shape[key_name]
            )
            copy_grads = torch.tensor(copy_grads).to(device)
            # copy the grads
            value.grad.data.copy_(copy_grads.data)
            pointer += len_grads


def _get_flat_grads(network):
    grads_shape = {}
    flat_grads = None
    for key_name, value in network.named_parameters():
        try:
            grads_shape[key_name] = value.grad.data.cpu().numpy().shape
        except:
            print("Cannot get grad of tensor {}".format(key_name))
            continue

        if flat_grads is None:
            flat_grads = value.grad.data.cpu().numpy().flatten()
        else:
            flat_grads = np.append(flat_grads, value.grad.data.cpu().numpy().flatten())
    return flat_grads, grads_shape


def soft_copy_network(target, source, tau):
    """Blends `target` and `source`: `target` = `tau` * `target` + `(1-tau)` * `source`."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.lerp_(source_param.data, 1 - tau)


def copy_network(target, source):
    """Copies network parameters: `target` = `source`."""
    soft_copy_network(target, source, 0)


def fig2tensor(draw_func):
    def decorate(*args, **kwargs):
        tmp = io.BytesIO()
        fig = draw_func(*args, **kwargs)
        fig.savefig(tmp, dpi=88)
        tmp.seek(0)
        fig.clf()
        return TF.to_tensor(PIL.Image.open(tmp))

    return decorate


def tensor2np(t):
    if isinstance(t, torch.Tensor):
        return t.clone().detach().cpu().numpy()
    else:
        return t


def tensor2img(tensor):
    if len(tensor.shape) == 4:
        assert tensor.shape[0] == 1
        tensor = tensor.squeeze(0)
    img = tensor.permute(1, 2, 0).detach().cpu().numpy()
    import cv2

    cv2.imwrite("tensor.png", img)


def dictlist_to_tensor(x, device):
    if isinstance(x, list):
        x = list2dict(x)

    return OrderedDict(
        [
            (k, torch.tensor(np.stack(v), dtype=torch.float32).to(device))
            for k, v in x.items()
        ]
    )


def to_tensor(x, device, dtype=torch.float):
    """Transfer a numpy array into a tensor."""
    if x is None:
        return x
    if isinstance(x, dict):
        return OrderedDict(
            [(k, torch.as_tensor(v, device=device, dtype=dtype)) for k, v in x.items()]
        )
    if isinstance(x, list):
        return [torch.as_tensor(v, device=device, dtype=dtype) for v in x]
    return torch.as_tensor(x, device=device, dtype=dtype)


def list2dict(rollout):
    ret = OrderedDict()
    for k in rollout[0].keys():
        ret[k] = []
    for transition in rollout:
        for k, v in transition.items():
            ret[k].append(v)
    return ret


def scale_dict_tensor(tensor, scalar):
    if isinstance(tensor, dict):
        return OrderedDict(
            [(k, scale_dict_tensor(tensor[k], scalar)) for k in tensor.keys()]
        )
    elif isinstance(tensor, list):
        return [scale_dict_tensor(tensor[i], scalar) for i in range(len(tensor))]
    else:
        return tensor * scalar


def space2tensor(batch_size, space, scalar, device):
    """Creates a tensor of shape `space` with value `scalar`."""
    if isinstance(space, gym.spaces.Dict):
        return OrderedDict(
            [
                (k, space2tensor(batch_size, s, scalar, device))
                for k, s in space.spaces.items()
            ]
        )
    elif isinstance(space, gym.spaces.Box):
        return scalar * torch.ones(batch_size + space.shape, device=device)
    elif isinstance(space, gym.spaces.Discrete):
        return scalar * torch.ones(batch_size + [1], device=device)


# From softlearning repo
def flatten(unflattened, parent_key="", separator="/"):
    items = []
    for k, v in unflattened.items():
        if separator in k:
            raise ValueError("Found separator ({}) from key ({})".format(separator, k))
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, dict) and v:
            items.extend(flatten(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))

    return OrderedDict(items)


# From softlearning repo
def unflatten(flattened, separator="."):
    result = {}
    for key, value in flattened.items():
        parts = key.split(separator)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return result


# from https://github.com/MishaLaskin/rad/blob/master/utils.py
def center_crop(img, out=84):
    """
    Args:
        imgs: np.array shape (C,H,W)
        out: output size (e.g. 84)
        returns np.array shape (1,C,H,W)
    """
    h, w = img.shape[1:]
    new_h, new_w = out, out

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    img = img[:, top : top + new_h, left : left + new_w]
    img = np.expand_dims(img, axis=0)
    return img


# from https://github.com/MishaLaskin/rad/blob/master/utils.py
def center_crop_images(image, out=84):
    """
    Args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array shape (B,C,H,W)
    """
    h, w = image.shape[2:]
    if isinstance(out, list):
        out_h, out_w = out
    else:
        out_h = out_w = out

    top = (h - out_h) // 2
    left = (w - out_w) // 2

    image = image[:, :, top : top + out_h, left : left + out_w]
    return image


# from https://github.com/MishaLaskin/rad/blob/master/data_augs.py
def random_crop(imgs, out=84):
    """
    Args:
        imgs: np.array shape (B,C,H,W)
        out: output size int (e.g. 84) or list (e.g. [84, 84])
        returns np.array
    """
    shape = imgs.shape
    imgs = imgs.reshape(-1, *shape[-3:])
    b, c, h, w = imgs.shape
    if isinstance(out, list):
        out_h, out_w = out[:2]
    else:
        out_h = out_w = out
    w1 = np.random.randint(0, w - out_w + 1, b)
    h1 = np.random.randint(0, h - out_h + 1, b)
    cropped = np.empty((b, c, out_h, out_w), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11 : h11 + out_h, w11 : w11 + out_w]
    cropped = cropped.reshape(*shape[:-3], c, out_h, out_w)
    return cropped


class RandomShiftsAug(nn.Module):
    """Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2 and https://github.com/nicklashansen/tdmpc
    """

    def __init__(self):
        super().__init__()
        # self.pad = int(cfg.img_size/21) if cfg.modality == 'pixels' else None
        self._pad = 4

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, *shape[-3:])
        x = x.permute(0, 3, 1, 2)
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self._pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self._pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self._pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(
            0, 2 * self._pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self._pad)
        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(*shape)
        return x


class RequiresGrad(object):
    def __init__(self, *model):
        self._model = model

    def __enter__(self):
        for m in self._model:
            m.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        for m in self._model:
            m.requires_grad_(requires_grad=False)


class AdamAMP(object):
    """Adam optimizer for automatic mixed precision."""

    def __init__(self, model, lr, weight_decay, grad_clip, device, use_amp):
        self._model = model
        self._optim = optim.Adam(
            model.parameters(), lr=lr, eps=1e-7, weight_decay=weight_decay
        )
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self._grad_clip = grad_clip
        self._device = device
        self._use_amp = use_amp

    def step(self, loss, retain_graph=False):
        self._optim.zero_grad()
        if self._use_amp:
            self._scaler.scale(loss).backward(retain_graph=retain_graph)
            self._scaler.unscale_(self._optim)
        else:
            loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip)
        sync_grad(self._model, self._device)

        if self._use_amp:
            self._scaler.step(self._optim)
            self._scaler.update()
        else:
            self._optim.step()

        return grad_norm

    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, state_dict):
        self._optim.load_state_dict(state_dict)

    @property
    def state(self):
        return self._optim.state


def check_memory_kill_switch(avail_thresh=10.0):
    """Kills program if available memory is below threshold to avoid memory overflows."""
    try:
        stat = psutil.virtual_memory()
        if stat.available * 100 / stat.total < avail_thresh:
            print(
                f"Current memory usage of {stat.percent}% surpasses threshold, killing program..."
            )
            # Avoid that all processes get killed at once
            time.sleep(10 * np.random.rand())
            if stat.available * 100 / stat.total < avail_thresh:
                exit(0)
    except FileNotFoundError:  # seems to happen infrequently
        pass


def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))
