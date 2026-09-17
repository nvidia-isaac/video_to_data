# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Misc functions, including distributed helpers, mostly from torchvision
"""
import glob
import math
import os
import time
import datetime
import random
from functools import partial
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Callable, Optional
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from accelerate import Accelerator
from learning.training.training_config import TrainTemporalRefinerConfig




@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    best_val: Optional[float] = None
    epoch_step: Optional[int] = None
    best_step: Optional[int] = None
    pending_validation_step: Optional[int] = None


def get_optimizer(cfg, model: torch.nn.Module, accelerator: Accelerator) -> torch.optim.Optimizer:
    """Gets optimizer from configs"""
    
    # Determine the learning rate
    if cfg.optimizer.scale_learning_rate_with_batch_size:
        lr = accelerator.state.num_processes * cfg.dataloader.batch_size * cfg.optimizer.lr
        print('lr = {ws} (num gpus) * {bs} (batch_size) * {blr} (base learning rate) = {lr}'.format(
            ws=accelerator.state.num_processes, bs=cfg.dataloader.batch_size, blr=cfg.optimizer.lr, lr=lr))
    else:  # scale base learning rate by batch size
        lr = cfg.optimizer.lr
        print('lr = {lr} (absolute learning rate)'.format(lr=lr))

    # Get optimizer parameters, excluding certain parameters from weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.optimizer.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # Construct optimizer
    if cfg.optimizer.type == 'torch':
        Optimizer: torch.optim.Optimizer = getattr(torch.optim, cfg.optimizer.name)
        optimizer = Optimizer(parameters, lr=lr, **cfg.optimizer.kwargs)
    elif cfg.optimizer.type == 'timm':
        from timm.optim import create_optimizer_v2
        optimizer = create_optimizer_v2(model_or_params=parameters, lr=lr, **cfg.optimizer.kwargs)
    elif cfg.optimizer.type == 'transformers':
        import transformers
        Optimizer: torch.optim.Optimizer = getattr(transformers, cfg.optimizer.name)
        optimizer = Optimizer(parameters, lr=lr, **cfg.optimizer.kwargs)
    else:
        raise NotImplementedError(f'Invalid optimizer configs: {cfg.optimizer}')

    return optimizer


def get_scheduler(cfg: TrainTemporalRefinerConfig, optimizer: torch.optim.Optimizer) -> Callable:
    """Gets scheduler from configs"""
    print("scheduler configs:", cfg.lr_scheduler)
    
    # Get scheduler
    if cfg.lr_scheduler.type == 'torch':
        Scheduler: torch.optim.lr_scheduler._LRScheduler = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.type)
        scheduler = Scheduler(optimizer=optimizer, **cfg.lr_scheduler.kwargs)
        if cfg.lr_scheduler.get('warmup', 0):
            from warmup_scheduler import GradualWarmupScheduler
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, 
                total_epoch=cfg.lr_scheduler.warmup, after_scheduler=scheduler)
    elif cfg.lr_scheduler.type == 'timm':
        from timm.scheduler import create_scheduler
        scheduler, _ = create_scheduler(optimizer=optimizer, args=cfg.lr_scheduler.kwargs)
    elif cfg.lr_scheduler.type == 'transformers':
        print(f'using scheduler with configs: {cfg.lr_scheduler.kwargs}')
        from transformers import get_scheduler # default: linear scheduler with warm up and linear decay
        scheduler = get_scheduler(optimizer=optimizer, **cfg.lr_scheduler.kwargs)
    elif cfg.lr_scheduler.type == 'cosine_floor':
        from learning.training.lr_scheduler import cosine_decay_with_floor_factor
        total_steps = int(cfg.lr_scheduler.kwargs.num_training_steps)
        scheduler_kwargs = cfg.lr_scheduler.kwargs
        end_lr = scheduler_kwargs.get('end_lr') if hasattr(scheduler_kwargs, 'get') else getattr(scheduler_kwargs, 'end_lr', None)
        end_lr_ratio = scheduler_kwargs.get('end_lr_ratio') if hasattr(scheduler_kwargs, 'get') else getattr(scheduler_kwargs, 'end_lr_ratio', None)
        lr_lambdas = [partial(cosine_decay_with_floor_factor, total_steps=total_steps, initial_lr=float(group['lr']), end_lr=end_lr, end_lr_ratio=end_lr_ratio) for group in optimizer.param_groups]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambdas)
    elif cfg.lr_scheduler.type == 'epoch_half_by':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **cfg.lr_scheduler.kwargs)
    elif cfg.lr_scheduler.type == 'step_half_by':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **cfg.lr_scheduler.kwargs)
    else:
        raise NotImplementedError(f'invalid scheduler configs: {cfg.lr_scheduler}')

    return scheduler


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, device='cuda'):
        """
        Warning: does not synchronize the deque!
        """
        if not using_distributed():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / max(self.count, 1)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value) if len(self.deque) > 0 else ""


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        n = kwargs.pop('n', 1)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n=n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self, device='cuda'):
        for meter in self.meters.values():
            meter.synchronize_between_processes(device=device)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{}  Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def resume_from_checkpoint(cfg: TrainTemporalRefinerConfig, model, optimizer=None, scheduler=None, model_ema=None):
    
    # Check if resuming training from a checkpoint
    if not cfg.checkpoint.resume:
        print('Starting training from scratch')
        return TrainState()

    # XH: find checkpiont path automatically
    if not os.path.isfile(cfg.checkpoint.resume):
        print(f"The given checkpoint path {cfg.checkpoint.resume} does not exist, trying to find one...")
        ckpt_file = os.path.join(cfg.run.code_dir_abs, f'outputs/{cfg.run.name}/single/checkpoint-latest.pth')
        if not os.path.isfile(ckpt_file):
            # just get the fist dir, for backward compatibility
            folders = sorted(glob.glob(os.path.join(cfg.run.code_dir_abs, f'outputs/{cfg.run.name}/2023-*')))
            assert len(folders) <= 1
            if len(folders) > 0:
                ckpt_file = os.path.join(folders[0], 'checkpoint-latest.pth')

        if os.path.isfile(ckpt_file):
            print(f"Found checkpoint at {ckpt_file}!")
            cfg.checkpoint.resume = ckpt_file
        else:
            print(f"No checkpoint found in outputs/{cfg.run.name}/single/!")
            return TrainState()

    # If resuming, load model state dict
    print(f'Loading checkpoint ({datetime.datetime.now()})')
    checkpoint = torch.load(cfg.checkpoint.resume, map_location='cpu')
    if 'model' in checkpoint:
        state_dict, key = checkpoint['model'], 'model'
    else:
        print("Warning: no model found in checkpoint!")
        state_dict, key = checkpoint, 'N/A'
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        print('Removed "module." from checkpoint state dict')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded model checkpoint key {key} from {cfg.checkpoint.resume}')
    if len(missing_keys):
        print(f' - Missing_keys: {missing_keys}')
    if len(unexpected_keys):
        print(f' - Unexpected_keys: {unexpected_keys}')
    # 298 missing, 328 unexpected! total 448 modules.
    print(f"{len(missing_keys)} missing, {len(unexpected_keys)} unexpected! total {len(model.state_dict().keys())} modules.")
    if 'step' in checkpoint:
        print("Number of trained steps:", checkpoint['step'])

    # TODO: implement better loading for fine tuning
    # Resume model ema
    if cfg.ema.use_ema:
        if checkpoint['model_ema']:
            model_ema.load_state_dict(checkpoint['model_ema'])
            print('Loaded model ema from checkpoint')
        else:
            model_ema.load_state_dict(model.parameters())
            print('No model ema in checkpoint; loaded current parameters into model')
    else:
        if 'model_ema' in checkpoint and checkpoint['model_ema']:
            print('Not using model ema, but model_ema found in checkpoint (you probably want to resume it!)')
        else:
            print('Not using model ema, and no model_ema found in checkpoint.')
            
    # Resume optimizer and/or training state
    train_state = TrainState()
    if 'train' in cfg.run.job:
        if cfg.checkpoint.resume_training:
            assert (
                cfg.checkpoint.resume_training_optimizer 
                or cfg.checkpoint.resume_training_scheduler 
                or cfg.checkpoint.resume_training_state
                or cfg.checkpoint.resume_training
            ), f'Invalid configs: {cfg.checkpoint}'
            if cfg.checkpoint.resume_training_optimizer:
                if 'optimizer' not in checkpoint:
                    print("Warning: not loading optimizer!")
                else:
                    assert 'optimizer' in checkpoint, f'Value not in {checkpoint.keys()}'
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print(f'Loaded optimizer from checkpoint')
            else:
                print(f'Did not load optimizer from checkpoint')
            if cfg.checkpoint.resume_training_scheduler:
                if 'scheduler' not in checkpoint:
                    print("Warning: not loading scheduler!")
                else:
                    assert 'scheduler' in checkpoint, f'Value not in {checkpoint.keys()}'
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    print(f'Loaded scheduler from checkpoint')
            else:
                print(f'Did not load scheduler from checkpoint')
            if cfg.checkpoint.resume_training_state:
                if 'steps' in checkpoint and 'step' not in checkpoint:  # fixes an old typo
                    checkpoint['step'] = checkpoint.pop('steps')
                assert {'epoch', 'step', 'best_val'}.issubset(set(checkpoint.keys()))
                epoch, step, best_val = checkpoint['epoch'] + 1, checkpoint['step'], checkpoint['best_val']
                train_state = TrainState(epoch=epoch, step=step, best_val=best_val)
                print(f'Resumed state from checkpoint: step {step}, epoch {epoch}, best_val {best_val}')
            else:
                print(f'Did not load train state from checkpoint')
        else:
            print('Did not resume optimizer, scheduler, or epoch from checkpoint')
    
    print(f'Finished loading checkpoint ({datetime.datetime.now()})')

    return train_state


def setup_distributed_print(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    from rich import print as __richprint__
    builtin_print = __richprint__  # __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def using_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if using_distributed() else 0


def set_seed(seed):
    rank = get_rank()
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if using_distributed():
        print(f'Seeding node {rank} with seed {seed}', force=True)
    else:
        print(f'Seeding node {rank} with seed {seed}')


def compute_grad_norm(parameters):
    total_norm = 0
    for p in parameters:
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
