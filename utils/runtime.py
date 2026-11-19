"""Random seeds, data loaders and single-GPU optimizer configuration."""
import random
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from datasets import build_dataset, collate_fn_crowd


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_data_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_optimizer(model, config):
    options = config["training"]

    groups = [
        {"params": [p for name, p in model.named_parameters() if "backbone" not in name and p.requires_grad]},
        {"params": [p for name, p in model.named_parameters() if "backbone" in name and p.requires_grad], "lr": options["lr_backbone"]},
    ]
    return torch.optim.Adam(groups, lr=options["lr"], weight_decay=options["adam_weight_decay"])


def build_loader(config, split):
    torch.multiprocessing.set_sharing_strategy("file_system")
    dataset = build_dataset(config, split)
    workers = config["training"]["num_workers"]
    if split == "train":
        generator = torch.Generator().manual_seed(config["seed"] + 10001)
        worker_generator = torch.Generator().manual_seed(config["seed"] + 20001)
        sampler = RandomSampler(dataset, generator=generator)
        batch_sampler = BatchSampler(sampler, config["training"]["batch_size"], drop_last=True)
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=workers,
                          collate_fn=collate_fn_crowd, worker_init_fn=seed_data_worker, generator=worker_generator)
    return DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset), drop_last=False,
                      collate_fn=partial(collate_fn_crowd, pad_block=config["evaluation"]["pad_block"]), num_workers=workers)
