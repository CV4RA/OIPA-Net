"""Q/V low-rank adaptation for the native PE attention projections."""
import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint


class QVLoRA(nn.Module):
    def __init__(self, width, rank, alpha):
        super().__init__()
        self.q_down = nn.Parameter(torch.randn(rank, width) * 0.02)
        self.v_down = nn.Parameter(torch.randn(rank, width) * 0.02)
        self.q_up = nn.Parameter(torch.zeros(width, rank))
        self.v_up = nn.Parameter(torch.zeros(width, rank))
        self.scale = alpha / rank

    def forward(self, weight):
        with torch.autocast(weight.device.type, enabled=False):
            q_delta = self.q_up @ self.q_down
            v_delta = self.v_up @ self.v_down
            delta = torch.cat((q_delta, torch.zeros_like(q_delta), v_delta))
        return weight + self.scale * delta.to(weight.dtype)


def checkpoint_forward(forward):
    def run(*args, **kwargs):
        if torch.is_grad_enabled():
            return checkpoint(forward, *args, use_reentrant=False, **kwargs)
        return forward(*args, **kwargs)

    return run


def attach_pe_lora(backbone, *, layers, rank, alpha, checkpoint_blocks):
    backbone.requires_grad_(False)
    model = backbone.model
    towers = (model.visual.transformer.resblocks, model.transformer.resblocks)
    if rank < 1 or alpha <= 0 or any(not 1 <= layers <= len(blocks) for blocks in towers):
        raise ValueError("Invalid PE LoRA rank, alpha or layer count")
    for blocks in towers:
        for block in blocks[-layers:]:
            attention = block.attn
            weight = attention.in_proj_weight
            width = weight.shape[1]
            if weight.shape != (3 * width, width):
                raise ValueError("Expected PE's fused Q/K/V weight")
            adapter = QVLoRA(width, rank, alpha).to(weight.device)
            parametrize.register_parametrization(attention, "in_proj_weight", adapter)
            if checkpoint_blocks:
                block.forward = checkpoint_forward(block.forward)


def trainable_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.named_parameters() if value.requires_grad}


@torch.no_grad()
def restore_trainable(module, state):
    parameters = {name: value for name, value in module.named_parameters() if value.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError("Adapter checkpoint parameter names differ")
    for name, value in parameters.items():
        if value.shape != state[name].shape:
            raise ValueError(f"Adapter checkpoint parameter shape differs: {name}")
        value.copy_(state[name])
