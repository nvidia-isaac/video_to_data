import torch


def forward_contact_head_fp32(contact_head, features):
    with torch.autocast(device_type=features.device.type, enabled=False):
        return contact_head(features.float()).mean(dim=1)
