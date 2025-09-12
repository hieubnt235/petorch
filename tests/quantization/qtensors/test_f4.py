# type: ignore

from math import sqrt
from typing import Literal
import ctypes as ct
import pytest
from PIL.ImageOps import scale
from torch import Tensor, float16, float32, bfloat16, nn
import torch

from petorch.quantization import F4QTensor, F4QConfig, F4QState


def devices_to_test() -> list[torch.device]:
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devs.append(torch.device(f"cuda:{i}"))
    return devs


available_devices = devices_to_test()
float_dtypes = [float16, float32, bfloat16]


def is_finite(t: Tensor) -> Tensor:
    return torch.all(torch.isfinite(t))


def abs_max(t: Tensor) -> int:
    return t.abs().max().item()


def similarity(ref: Tensor, t: Tensor) -> Tensor:
    """
    (AB) / (|A|^2 + |B|^2)
    Args:
        ref:
        t:

    Returns:
        [0,1]

    """
    # Scale for not out of range.
    scale = ref.numel()
    sqrt_scale = sqrt(float(scale))

    scaled_dot = (ref.view(-1) * t.view(-1) / scale).sum()
    # dot = torch.matmul(ref.view(-1), t.view(-1))
    scaled_norm = (torch.norm(ref) / sqrt_scale).pow(2) + (
        torch.norm(t) / sqrt_scale
    ).pow(2)

    r: Tensor = 2 * scaled_dot / scaled_norm
    assert is_finite(
        r
    ), f"{r} {is_finite(ref)} {is_finite(t)} {[abs_max(t) for t in [scaled_dot, scaled_norm]] }"
    return r


atol = 5e-3
rtol = 5e-3


def assert_similarity(
    ta: Tensor,
    tb: Tensor,
    *,
    exclude: list[Literal["device", "dtype", "requires_grad", "sim"]] = None,
):
    """
    Args:
        ta: Hp float tensor
        tb: Hp float tensor
        exclude: things that not including to test
    """
    exclude = exclude or []
    for attr in ["device", "dtype", "requires_grad"]:
        if not attr in exclude:
            assert (a := getattr(ta, attr)) == (b := getattr(tb, attr)), f"{a}, {b}"

    if not "sim" in exclude:
        sim = similarity(ta, tb)
        target = torch.tensor(
            1.0, dtype=sim.dtype, device=sim.device, requires_grad=sim.requires_grad
        )

        assert torch.allclose(
            sim,
            target,
            rtol=rtol,
            atol=atol,
        ), (
            1.0 - sim.item()
        )


def make_nf4(tensor: Tensor, config: F4QConfig | None = None) -> F4QTensor:
    config = config or F4QConfig(quant_type="nf4")
    nf4 = F4QTensor.from_high_precision(tensor, config)
    assert torch.all(torch.isfinite(nf4.get_high_precision()))
    for k, ts in nf4.iter_tensors():
        assert type(ts) == Tensor
        assert torch.all(torch.isfinite(ts))
    return nf4


@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("requires_grad", [True, False])
def test_original_quantized_similar_attributes_and_values(device, dtype, requires_grad):
    # Relative large tensor
    t = torch.randn([256, 256], device=device, dtype=dtype, requires_grad=requires_grad)
    nf4 = make_nf4(t)
    deq = nf4.get_high_precision()
    assert_similarity(t, deq)
    assert_similarity(nf4, deq, exclude=["sim"])

    # Relative small tensor
    # TODO: lower than 16x16 cause Segmentation Error. No clear reason.
    t = torch.randn([16, 16], device=device, dtype=dtype, requires_grad=requires_grad)
    nf4 = make_nf4(t)
    deq = nf4.get_high_precision()
    assert_similarity(t, deq)
    assert_similarity(nf4, deq, exclude=["sim"])


@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("requires_grad", [True, False])
def test_change_attributes_with_to_cpu_cuda(device, dtype, requires_grad):
    t = torch.randn([16, 16])
    nf4 = make_nf4(t)

    ref_attrs = [device, dtype, requires_grad]
    attrs_str = ["device", "dtype", "requires_grad"]
    nf4 = nf4.to(dtype=dtype, device=device).requires_grad_(requires_grad)

    for attr, ref in zip(attrs_str, ref_attrs):
        assert (a := getattr(nf4.get_high_precision(), attr)) == ref, f"{a},{ref}"

    nf4 = nf4.cuda()
    assert nf4.get_high_precision().device.type == "cuda"
    nf4 = nf4.cpu()
    assert nf4.get_high_precision().device.type == "cpu"


from torchao.dtypes import NF4Tensor


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("requires_grad", [True, False])
def test_qtensor_copy_semantics(device, dtype, requires_grad):
    # --- base tensor
    t = torch.randn([16, 16], dtype=dtype, device=device, requires_grad=requires_grad)
    q = make_nf4(t)
    q_hp = q.get_high_precision()

    #
    # clone
    q_clone_hp = q.clone().get_high_precision()
    assert_similarity(q_clone_hp, q_hp)
    assert_similarity(q_clone_hp, t)

    # detach
    q_det = q.detach()
    q_det_hp = q.detach().get_high_precision()
    assert not q_det.requires_grad and not q_det_hp.requires_grad

    assert_similarity(q_det_hp, q_hp, exclude=["requires_grad"])
    assert_similarity(q_det_hp, t, exclude=["requires_grad"])

    # copy_ (torch <- QTensor)

    # --- Change dtype the copy_ copy values only, not device or dtype.
    q2 = q.to(device="cpu", dtype=bfloat16)

    # --- Support both methods.
    t2 = torch.empty_like(t)
    t3 = torch.empty_like(t)
    t2.copy_(q2)
    t3.copy_(q2.get_high_precision())
    assert_similarity(t2, t)
    assert_similarity(t3, t)
    # #
    # # # copy_ (QTensor <- torch)
    # q3 = make_nf4(torch.empty_like(t))
    q3 = make_nf4(torch.zeros_like(t))
    # q3.copy_(t)
    # assert isinstance(q3, F4QTensor)
    # assert_similarity(q3.get_high_precision(), t)
    #
    #
    # # copy_ (QTensor <- QTensor)
    # q4 = make_nf4(torch.empty_like(t))
    # q4.copy_(q2)
    # # --- q2 is the one that changes the device and dtype from q; now I assert with q.
    # assert_similarity(q4.get_high_precision(), q.get_high_precision())

    # ensure gradients flow through clone but not detach
    # if requires_grad:
    #
    #     q_clone = q.clone()
    #     assert isinstance(q_clone, F4QTensor)
    #     out_clone = (q_clone * 2).sum()
    #     out_clone.backward()
    #     assert q_clone.grad is not None
    #
    #     q_det = q.detach()
    #     assert isinstance(q_det, F4QTensor)
    #     out_det = (q_det * 2).sum()
    #     out_det.backward()
    #     # detached → no gradient should propagate
    #     assert q_det.grad is None


from torchao.dtypes import NF4Tensor

# @pytest.mark.parametrize("device", available_devices)
# @pytest.mark.parametrize("dtype", float_dtypes)
# @pytest.mark.parametrize("requires_grad", [True, False])
# @pytest.mark.parametrize("quantize_weight", [True, False])
# @pytest.mark.parametrize("quantize_bias", [True, False])
# def test_linear(device, dtype, requires_grad, quantize_weight, quantize_bias):
#     in_features, out_features, batch_size = 16, 32, 4
#     x = torch.randn(
#         batch_size, in_features, dtype=dtype, device=device, requires_grad=requires_grad
#     )
#
#     ref_linear = (
#         nn.Linear(in_features, out_features).to(device=device, dtype=dtype).train()
#     )
#
#     q_linear = (
#         nn.Linear(in_features, out_features).to(device=device, dtype=dtype).train()
#     )
#
#     # Copy reference params
#     with torch.no_grad():
#         q_linear.weight.copy_(ref_linear.weight)
#         if ref_linear.bias is not None:
#             q_linear.bias.copy_(ref_linear.bias)
#
#     # Quantize
#     if quantize_weight:
#         q_linear.weight = nn.Parameter(make_nf4(q_linear.weight))
#     if quantize_bias and q_linear.bias is not None:
#         q_linear.bias = nn.Parameter(make_nf4(q_linear.bias))
#
#     # --- Forward
#     y_ref = ref_linear(x)
#     y_q = q_linear(x)
#
#     simi = similarity(y_q, y_ref.detach())
#     target = torch.tensor(1.0, dtype=dtype, device=device, requires_grad=requires_grad)
#     assert torch.allclose(
#         simi, target, rtol=1e-3, atol=1e-3
#     ), f"Forward mismatch: similarity={simi.item():.6f}"
#
#     # --- Backward (if requires_grad)
#     if requires_grad:
#         grad = torch.randn_like(y_ref)
#         y_ref.backward(grad)
#         y_q.backward(grad)
#
#         # Check gradients of input
#         simi_in = similarity(x.grad, x.grad)  # sanity check, should be identical types
#         assert simi_in.item() >= 0.99, f"Grad input mismatch {simi_in.item()}"
#
#         # Check gradients of weight and bias
#         if quantize_weight:
#             simi_w = similarity(q_linear.weight.grad, ref_linear.weight.grad)
#             assert torch.allclose(simi_w, target, rtol=1e-3, atol=1e-3)
#         else:
#             assert torch.allclose(
#                 q_linear.weight.grad, ref_linear.weight.grad, rtol=1e-3, atol=1e-3
#             )
#
#         if q_linear.bias is not None:
#             if quantize_bias:
#                 simi_b = similarity(q_linear.bias.grad, ref_linear.bias.grad)
#                 assert torch.allclose(simi_b, target, rtol=1e-3, atol=1e-3)
#             else:
#                 assert torch.allclose(
#                     q_linear.bias.grad, ref_linear.bias.grad, rtol=1e-3, atol=1e-3
#                 )
#
#     # --- Freeze (eval mode, no dropout/BN but ensure still works)
#     q_linear.eval()
#     with torch.no_grad():
#         y_eval = q_linear(x)
#         y_ref_eval = ref_linear.eval()(x)
#     simi_eval = similarity(y_eval, y_ref_eval)
#     assert torch.allclose(simi_eval, target, rtol=1e-3, atol=1e-3)


def test_permute():
    pass


def test_add():
    pass


def test_mul():
    pass
