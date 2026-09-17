import math


DEFAULT_COSINE_END_LR_RATIO = 0.01


def resolve_cosine_end_lr(initial_lr, end_lr=None, end_lr_ratio=None):
    initial_lr = float(initial_lr)
    if initial_lr <= 0:
        raise ValueError(f"initial_lr must be positive, got {initial_lr}")
    if end_lr is not None and end_lr_ratio is not None:
        raise ValueError("end_lr and end_lr_ratio are mutually exclusive")
    if end_lr is None:
        end_lr_ratio = DEFAULT_COSINE_END_LR_RATIO if end_lr_ratio is None else float(end_lr_ratio)
        if end_lr_ratio < 0 or end_lr_ratio > 1:
            raise ValueError(f"end_lr_ratio must be in [0, 1], got {end_lr_ratio}")
        return initial_lr * end_lr_ratio
    end_lr = float(end_lr)
    if end_lr < 0 or end_lr > initial_lr:
        raise ValueError(f"end_lr must be in [0, initial_lr], got end_lr={end_lr}, initial_lr={initial_lr}")
    return end_lr


def cosine_decay_with_floor_factor(current_step, total_steps, initial_lr, end_lr=None, end_lr_ratio=None):
    total_steps = int(total_steps)
    initial_lr = float(initial_lr)
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    end_lr = resolve_cosine_end_lr(initial_lr, end_lr=end_lr, end_lr_ratio=end_lr_ratio)
    progress = min(max(float(current_step), 0.0) / total_steps, 1.0)
    current_lr = end_lr + 0.5 * (initial_lr - end_lr) * (1.0 + math.cos(math.pi * progress))
    return current_lr / initial_lr


def rebase_lambda_scheduler_to_step(scheduler, global_step):
    global_step = int(global_step)
    if global_step < 0:
        raise ValueError(f"global_step must be non-negative, got {global_step}")
    if len(scheduler.base_lrs) != len(scheduler.lr_lambdas) or len(scheduler.base_lrs) != len(scheduler.optimizer.param_groups):
        raise ValueError("Lambda scheduler base learning rates, lambdas, and optimizer parameter groups must have equal lengths")
    learning_rates = [base_lr * lr_lambda(global_step) for base_lr, lr_lambda in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
    scheduler.last_epoch = global_step
    scheduler._step_count = global_step + 1
    for parameter_group, learning_rate in zip(scheduler.optimizer.param_groups, learning_rates):
        parameter_group["lr"] = learning_rate
    scheduler._last_lr = learning_rates
