"""Weight-only quantization (Stage 8) on random layers — no model."""

import pytest
import torch
from torch import nn

from tiny_engine.quantization import QuantLinear, pack_int4, unpack_int4


def test_int4_pack_roundtrip():
    q = torch.randint(-8, 8, (5, 16), dtype=torch.int8)
    packed = pack_int4(q)
    assert packed.dtype == torch.uint8 and packed.shape == (5, 8)
    assert torch.equal(unpack_int4(packed), q)


@pytest.mark.parametrize("method,max_rel_err,ratio", [("int8", 0.015, 0.52), ("int4", 0.15, 0.27), ("fp8", 0.06, 0.52)])
def test_quant_linear_close_to_original(method, max_rel_err, ratio):
    if method == "fp8" and not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("no float8 in this torch")
    torch.manual_seed(0)
    lin = nn.Linear(256, 64, bias=True).to(torch.bfloat16)
    qlin = QuantLinear.from_linear(lin, method, group_size=128)
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    ref, out = lin(x).float(), qlin(x).float()
    rel = (out - ref).norm() / ref.norm()
    assert rel < max_rel_err, f"{method}: relative error {rel:.4f}"
    w_bytes = lin.weight.numel() * lin.weight.element_size()
    q_bytes = qlin.qweight.numel() * qlin.qweight.element_size() + qlin.scales.numel() * qlin.scales.element_size()
    assert q_bytes <= w_bytes * ratio + 1, f"{method}: {q_bytes} vs {w_bytes} bytes"


def test_int8_scales_are_per_output_channel():
    lin = nn.Linear(32, 8, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.arange(8, dtype=torch.float32)[:, None].expand(8, 32) + 1)
    q = QuantLinear.from_linear(lin, "int8")
    assert q.scales.shape == (8, 1)
    torch.testing.assert_close(q.dequantize(), lin.weight, atol=1e-2, rtol=1e-2)
