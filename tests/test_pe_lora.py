from types import SimpleNamespace

import torch
from torch import nn

from place_retrieval.pe_lora import attach_pe_lora, restore_trainable, trainable_state


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)

    def forward(self, value, attn_mask=None):
        return self.attn(value, value, value, need_weights=False, attn_mask=attn_mask)[0]


class TinyPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = AttentionBlock()
        self.text = AttentionBlock()
        self.model = SimpleNamespace(
            visual=SimpleNamespace(transformer=SimpleNamespace(resblocks=[self.vision])),
            transformer=SimpleNamespace(resblocks=[self.text]),
        )


def test_qv_adapters_keep_initial_output_and_restore_after_checkpointed_backward():
    torch.manual_seed(42)
    model = TinyPE()
    values = torch.randn(2, 3, 8)
    original = model.vision(values).detach()
    attach_pe_lora(model, layers=1, rank=2, alpha=4, checkpoint_blocks=True)
    torch.testing.assert_close(model.vision(values), original, rtol=0, atol=0)
    (model.vision(values).square().mean() + model.text(values).square().mean()).backward()
    for block in (model.vision, model.text):
        weights = block.attn.parametrizations.in_proj_weight
        assert weights.original.grad is None
        assert weights[0].q_up.grad.norm() > 0 and weights[0].v_up.grad.norm() > 0
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    optimizer.step()
    state = trainable_state(model)
    expected = model.vision(values).detach()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    restore_trainable(model, state)
    torch.testing.assert_close(model.vision(values), expected, rtol=0, atol=0)
