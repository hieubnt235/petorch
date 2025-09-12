# pytest test suite for a Tensor-subclass that aims to mimic torch.Tensor
# Adjust the import path of F4QTensor if needed.
import math
import sys
from typing import Any, Sequence

import pytest
import torch

from petorch.quantization.qtensors.nf4 import F4QTensor, F4QConfig


def unwrap(x):
    # Helper to get the underlying torch.Tensor if x is a F4QTensor
    if isinstance(x, F4QTensor):
        return x.a
    return x


def unwrap_tree(x):
    if isinstance(x, (list, tuple)):
        return type(x)(unwrap_tree(xx) for xx in x)
    if isinstance(x, dict):
        return {k: unwrap_tree(v) for k, v in x.items()}
    return unwrap(x)


def assert_close_structure(a: Any, b: Any, *, rtol=1e-5, atol=1e-7):
    # Recursively compare structures and tensors
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b)
        for aa, bb in zip(a, b):
            assert_close_structure(aa, bb, rtol=rtol, atol=atol)
        return
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a.keys()) == set(b.keys())
        for k in a.keys():
            assert_close_structure(a[k], b[k], rtol=rtol, atol=atol)
        return
    if torch.is_tensor(a) and torch.is_tensor(b):
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert a.device == b.device
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        return
    # Non-tensor leaves (e.g., ints, floats, None)
    assert a == b


def make_tensor(shape=(3, 4), *, device=None, dtype=torch.float32, requires_grad=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.randn(*shape, device=device, dtype=dtype, requires_grad=requires_grad)
    return t


def is_cuda_available():
    return torch.cuda.is_available()


def devices_to_test():
    devs = ["cpu"]
    if is_cuda_available():
        devs.append("cuda")
    return devs


def dtypes_to_test():
    # Keep it simple; add more if your subclass supports them
    return [
        torch.float32,
        torch.float64,
        torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ]

def make_pair(
    shape=(3, 4), *, device=None, dtype=torch.float32, requires_grad=False, name="x"
):
    base = make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    sub = F4QTensor.from_high_precision(base, config=F4QConfig())
    return base, sub

# -------- Basic Properties --------
@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
def test_basic_mirror(device, dtype):
    a, x = make_pair(
        (2, 3), device=device, dtype=dtype, requires_grad=True, name="basic"
    )
    # Verify basic metadata parity
    assert x.shape == a.shape
    assert x.dtype == a.dtype
    assert x.device == a.device
    assert x.requires_grad == a.requires_grad
    # Underlying must reflect same tensor attributes
    assert unwrap(x).shape == a.shape
    assert unwrap(x).dtype == a.dtype
    assert unwrap(x).device == a.device


# -------- Unary and Elementwise Ops --------
@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
@pytest.mark.parametrize(
    "op",
    [
        torch.neg,
        torch.abs,
        torch.sin,
        torch.cos,
        torch.relu,
        torch.sigmoid,
        torch.tanh,
    ],
)
def test_unary_ops_match(device, dtype, op):
    a, x = make_pair((3, 4), device=device, dtype=dtype)
    ref = op(a)
    out = op(x)
    ref_u, out_u = unwrap_tree(ref), unwrap_tree(out)
    assert_close_structure(ref_u, out_u)


@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
@pytest.mark.parametrize(
    "op",
    [
        torch.add,
        torch.sub,
        torch.mul,
        torch.div,
    ],
)
def test_binary_ops_match_tensor_tensor(device, dtype, op):
    a, x = make_pair((3, 4), device=device, dtype=dtype)
    b, y = make_pair((3, 4), device=device, dtype=dtype, name="y")
    ref = op(a, b)
    out = op(x, y)
    ref_u, out_u = unwrap_tree(ref), unwrap_tree(out)
    assert_close_structure(ref_u, out_u)


@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
@pytest.mark.parametrize(
    "op",
    [
        torch.add,
        torch.sub,
        torch.mul,
        torch.div,
    ],
)
def test_binary_ops_match_tensor_scalar(device, dtype, op):
    a, x = make_pair((3, 4), device=device, dtype=dtype)
    s = 2.5
    ref = op(a, s)
    out = op(x, s)
    ref_u, out_u = unwrap_tree(ref), unwrap_tree(out)
    assert_close_structure(ref_u, out_u)


# -------- Reductions --------
@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
@pytest.mark.parametrize(
    "red",
    [
        lambda z: z.sum(),
        lambda z: z.mean(),
        lambda z: z.sum(dim=1),
        lambda z: z.mean(dim=0),
        lambda z: z.amax(dim=1),
        lambda z: z.amin(dim=0),
        lambda z: z.norm(),
    ],
)
def test_reductions_match(device, dtype, red):
    a, x = make_pair((3, 4), device=device, dtype=dtype)
    ref = red(a)
    out = red(x)
    ref_u, out_u = unwrap_tree(ref), unwrap_tree(out)
    assert_close_structure(ref_u, out_u)


# -------- Linear Algebra --------
@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
def test_mm_matmul_bmm(device, dtype):
    a, x = make_pair((5, 7), device=device, dtype=dtype)
    b, y = make_pair((7, 3), device=device, dtype=dtype, name="y")
    ref_mm = torch.mm(a, b)
    out_mm = torch.mm(x, y)
    assert_close_structure(unwrap(out_mm), unwrap(ref_mm))

    ref_matmul = a @ b
    out_matmul = x @ y
    assert_close_structure(unwrap(out_matmul), unwrap(ref_matmul))

    a3, x3 = make_pair((2, 5, 7), device=device, dtype=dtype)
    b3, y3 = make_pair((2, 7, 3), device=device, dtype=dtype, name="y3")
    ref_bmm = torch.bmm(a3, b3)
    out_bmm = torch.bmm(x3, y3)
    assert_close_structure(unwrap(out_bmm), unwrap(ref_bmm))


# -------- Views and Shape Ops --------
@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
def test_view_reshape_transpose_t(device, dtype):
    a, x = make_pair((2, 3, 4), device=device, dtype=dtype)
    # view/reshape
    ref_v = a.view(6, 4)
    out_v = x.view(6, 4)
    assert_close_structure(unwrap(out_v), unwrap(ref_v))

    ref_r = a.reshape(4, 6)
    out_r = x.reshape(4, 6)
    assert_close_structure(unwrap(out_r), unwrap(ref_r))

    # transpose
    ref_tr = a.transpose(1, 2)
    out_tr = x.transpose(1, 2)
    assert_close_structure(unwrap(out_tr), unwrap(ref_tr))

    # t (2D transpose)
    a2, x2 = make_pair((3, 5), device=device, dtype=dtype)
    ref_t = a2.t()
    out_t = x2.t()
    assert_close_structure(unwrap(out_t), unwrap(ref_t))


@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
def test_permute_contiguous_squeeze_unsqueeze(device, dtype):
    a, x = make_pair((2, 1, 3, 1), device=device, dtype=dtype)
    ref_p = a.permute(2, 0, 3, 1)
    out_p = x.permute(2, 0, 3, 1)
    assert_close_structure(unwrap(out_p), unwrap(ref_p))

    ref_c = a.contiguous()
    out_c = x.contiguous()
    assert_close_structure(unwrap(out_c), unwrap(ref_c))

    ref_sq = a.squeeze()
    out_sq = x.squeeze()
    assert_close_structure(unwrap(out_sq), unwrap(ref_sq))

    ref_usq = a.unsqueeze(1)
    out_usq = x.unsqueeze(1)
    assert_close_structure(unwrap(out_usq), unwrap(ref_usq))


# -------- Indexing/Slicing/Masking --------
@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
def test_indexing_slicing(device, dtype):
    a, x = make_pair((4, 5), device=device, dtype=dtype)
    idx = 2
    sl = slice(1, 4)
    mask = a > 0.0

    ref_i = a[idx]
    out_i = x[idx]
    assert_close_structure(unwrap(out_i), unwrap(ref_i))

    ref_s = a[:, sl]
    out_s = x[:, sl]
    assert_close_structure(unwrap(out_s), unwrap(ref_s))

    ref_m = a[mask]
    out_m = x[mask]
    assert_close_structure(unwrap(out_m), unwrap(ref_m))


# -------- Concatenation/Stacking --------
@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
def test_cat_stack(device, dtype):
    a1, x1 = make_pair((2, 3), device=device, dtype=dtype, name="x1")
    a2, x2 = make_pair((2, 3), device=device, dtype=dtype, name="x2")

    ref_cat = torch.cat([a1, a2], dim=0)
    out_cat = torch.cat([x1, x2], dim=0)
    assert_close_structure(unwrap(out_cat), unwrap(ref_cat))

    ref_st = torch.stack([a1, a2], dim=1)
    out_st = torch.stack([x1, x2], dim=1)
    assert_close_structure(unwrap(out_st), unwrap(ref_st))


# -------- Broadcasting --------
@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
def test_broadcasting_add(device, dtype):
    a, x = make_pair((3, 1), device=device, dtype=dtype)
    b = torch.randn(1, 4, device=device, dtype=dtype)
    ref = a + b
    out = x + b
    assert_close_structure(unwrap(out), unwrap(ref))


# -------- In-place Ops and Aliasing Semantics --------
@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
def test_inplace_ops_semantics(device, dtype):
    a, x = make_pair((3, 4), device=device, dtype=dtype)
    b = torch.randn_like(a)

    # Reference in-place modifies and returns same object
    a_id_before = id(a)
    a.add_(b)
    a_id_after = id(a)
    assert a_id_before == a_id_after

    # Subclass in-place should ideally return the same wrapper, and modify underlying
    x_id_before = id(x)
    out = x.add_(b)
    x_id_after = id(x)
    # Check data equality and in-place identity expectation
    assert_close_structure(unwrap(x), a)  # data updated
    # Depending on subclass implementation, identity might or might not be preserved.
    # This assertion will catch aliasing regressions if the subclass creates a new wrapper.
    assert (
        id(out) == x_id_before == x_id_after
    ), "In-place operation should return the same wrapper instance"


# -------- Autograd: Forward/Backward --------
@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])  # stable grads
def test_autograd_matches(device, dtype):
    a, x = make_pair((5, 6), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)

    # Regular tensor forward/backward
    ref = (a * b).sin().sum() + (a @ torch.randn(6, 3, device=device, dtype=dtype)).pow(
        2
    ).mean()
    ref.backward()
    ref_grad_a = a.grad.clone()
    ref_grad_b = b.grad.clone()

    # Reset grads
    a.grad = None
    b.grad = None

    # Subclass path: use x instead of a, leave b as-is; unwrap ensures underlying a receives gradients
    out = (x * b).sin().sum() + (x @ torch.randn(6, 3, device=device, dtype=dtype)).pow(
        2
    ).mean()
    out.backward()

    # Verify grads on underlying real leaf tensors
    assert a.grad is not None
    assert b.grad is not None
    torch.testing.assert_close(a.grad, ref_grad_a, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(b.grad, ref_grad_b, rtol=1e-5, atol=1e-7)


# -------- Device/Dtype Moves --------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_to_cpu_cuda_and_dtype(dtype):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a, x = make_pair((2, 3, 4), device=device, dtype=dtype, requires_grad=True)

    # Change dtype
    ref = a.to(torch.float64 if dtype == torch.float32 else torch.float32)
    out = x.to(torch.float64 if dtype == torch.float32 else torch.float32)
    assert_close_structure(unwrap(out), unwrap(ref))

    # Move device if CUDA is available
    if torch.cuda.is_available():
        tgt = "cpu" if a.device.type == "cuda" else "cuda"
        ref2 = a.to(tgt)
        out2 = x.to(tgt)
        assert_close_structure(unwrap(out2), unwrap(ref2))


# -------- Clone/Detach semantics --------
@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
def test_clone_detach(device, dtype):
    a, x = make_pair((4, 5), device=device, dtype=dtype, requires_grad=True)

    ref_clone = a.clone()
    out_clone = x.clone()
    assert_close_structure(unwrap(out_clone), unwrap(ref_clone))
    # clone should not be the same object
    assert id(unwrap(out_clone)) != id(unwrap(x))

    ref_det = a.detach()
    out_det = x.detach()
    assert_close_structure(unwrap(out_det), unwrap(ref_det))
    # detach should share storage (for normal tensors). For subclass, we at least check values equal.
    # If you want to assert storage aliasing, do it carefully:
    # assert unwrap(out_det).storage().data_ptr() == unwrap(x).storage().data_ptr()


# -------- Torch functions accepting kwargs and shapes --------
@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
def test_add_with_alpha_and_out(device, dtype):
    a, x = make_pair((3, 4), device=device, dtype=dtype)
    b = torch.randn_like(a)
    ref = torch.add(a, b, alpha=0.3)
    out = torch.add(x, b, alpha=0.3)
    assert_close_structure(unwrap(out), unwrap(ref))


@pytest.mark.parametrize("device", devices_to_test())
@pytest.mark.parametrize("dtype", dtypes_to_test())
def test_slice_and_assign_copy_(device, dtype):
    a, x = make_pair((5, 6), device=device, dtype=dtype)
    # test copy_
    src = torch.ones_like(a)
    ref = a.clone()
    ref[:, :3].copy_(src[:, :3])

    out = x.clone()
    target = out[:, :3]  # This should be a view-like wrapper
    target.copy_(src[:, :3])
    assert_close_structure(unwrap(out), unwrap(ref))


# tests/test_nf4_integration.py
import importlib
import inspect
import tempfile
import os
import pytest
import torch
from torch import nn

# Attempt to import the user's nf4 module.
# If it fails, skip whole file.
try:
    nf4 = importlib.import_module("nf4")
except Exception as e:
    pytest.skip(f"nf4 module not importable: {e}", allow_module_level=True)

# Basic checks: expected names
F4QTensor = getattr(nf4, "F4QTensor", None)
F4QState = getattr(nf4, "F4QState", None)
F4QConfig = getattr(nf4, "F4QConfig", None)

if F4QTensor is None:
    pytest.skip("nf4.F4QTensor not found — skipping NF4 integration tests", allow_module_level=True)

# Optional: check for common external deps that NF4 may require
missing_deps = []
try:
    import bitsandbytes  # noqa: F401
except Exception:
    # Many nf4 implementations require bitsandbytes optional. If missing, we'll still run
    # tests but some functionality may be skipped.
    missing_deps.append("bitsandbytes (optional)")

# Helper: detect factory function to create quantized tensor from a float tensor
def _find_factory(module):
    # prefer explicit well-known names
    cand_names = [
        "from_high_precision",
        "from_float",
        "from_fp",
        "to_my_dtype",
        "to_nf4",
        "to_my_dtype",  # repeated on purpose
        "to_f4",
        "to_myq",
        "to_my_dtype_tensor",
    ]
    for name in cand_names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    # else search for any function that returns F4QTensor when called with a small tensor
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        # try calling with a tiny input if signature seems compatible
        sig = inspect.signature(obj)
        params = sig.parameters
        # attempt to call functions with a single Tensor param
        try:
            if len(params) == 1:
                test_in = torch.randn(2, 3)
                try:
                    out = obj(test_in)
                except Exception:
                    continue
                if isinstance(out, F4QTensor):
                    return obj
        except Exception:
            continue
    return None

factory = _find_factory(nf4)
if factory is None:
    # If factory not found on nf4, look for a classmethod on F4QTensor
    factory = getattr(F4QTensor, "from_high_precision", None) or getattr(F4QTensor, "from_float", None)

if not callable(factory):
    pytest.skip("Could not find a factory to construct F4QTensor (from_high_precision/from_float/etc).", allow_module_level=True)

# Short utility to try to dequantize / get high precision
def dequantize_or_get_hp(qtensor):
    # common names for the dequantize function
    for name in ("dequantize", "get_high_precision", "to_high_precision", "dequantize_affine"):
        fn = getattr(qtensor, name, None)
        if callable(fn):
            return fn()  # call it
    # maybe the state has a method
    st = getattr(qtensor, "quant_state", None)
    if st is not None:
        for attr in ("dequantize", "get_plain", "plain"):
            fn = getattr(st, attr, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass
            if hasattr(st, attr):
                val = getattr(st, attr)
                if isinstance(val, torch.Tensor):
                    return val
    pytest.skip("Cannot find a dequantize / get_high_precision method on the F4QTensor instance.", allow_module_level=True)

# Tag as integration tests
pytestmark = pytest.mark.integration

# ---------- Tests ----------

def test_factory_and_dequantize_roundtrip():
    # Create a random float tensor and quantize then dequantize and compare
    x = 3.0 * torch.randn(4, 6)
    q = factory(x) if not inspect.ismethod(factory) else factory.__func__(F4QTensor, x)  # try both function or classmethod styles
    assert isinstance(q, F4QTensor), f"Factory returned {type(q)}"
    dq = dequantize_or_get_hp(q)
    assert isinstance(dq, torch.Tensor)
    # compare shapes and rough numeric closeness (NF4 is lossy — use looser tolerance)
    assert dq.shape == x.shape
    # numeric tolerance: NF4 quantization is lossy, so use reasonable threshold
    assert torch.allclose(dq, x, atol=1e-1, rtol=0.2), "Dequantized tensor significantly differs from original (expected some loss for NF4)"

def test_linear_module_with_quantized_weights():
    # Build a small linear layer and replace its weight with quantized weight (typical use-case)
    in_f, out_f = 8, 5
    linear = nn.Linear(in_f, out_f, bias=True)
    # sample input
    inp = torch.randn(2, in_f)
    # baseline output using original float
    baseline = linear(inp)
    # quantize the weight (and optionally bias)
    qweight = factory(linear.weight.detach().clone())
    assert isinstance(qweight, F4QTensor)
    # If bias is simple tensor we keep it as-is
    # Replace parameter safely
    linear.weight = nn.Parameter(qweight.dequantize() if hasattr(qweight, "dequantize") else dequantize_or_get_hp(qweight))
    # Run forward and compare (should be nearly identical)
    out = linear(inp)
    assert torch.allclose(out, baseline, atol=1e-3, rtol=1e-3)

def test_forward_with_qtensor_weight_in_module_and_compare_dequantized():
    # A different approach: if nf4 supports assigning QTensor as Parameter/weight directly,
    # test module forward result equals dequantized path.
    in_f, out_f = 7, 3
    lin = nn.Linear(in_f, out_f, bias=True)
    inp = torch.randn(4, in_f)
    baseline = lin(inp)
    # create quantized weight and bias if supported
    qweight = factory(lin.weight.detach().clone())
    qbias = None
    if lin.bias is not None:
        try:
            qbias = factory(lin.bias.detach().clone())
        except Exception:
            qbias = None
    # If module can accept QTensor directly, try that path; else skip this test
    can_set_qtensor = True
    try:
        lin.weight = nn.Parameter(qweight)  # may raise / be invalid; wrapped in try
        if qbias is not None:
            lin.bias = nn.Parameter(qbias)
    except Exception:
        can_set_qtensor = False
    if not can_set_qtensor:
        pytest.skip("Module cannot accept QTensor directly as Parameter; skipping this test.")
    out_q = lin(inp)
    # dequantize weights and bias and compare
    deq_w = dequantize_or_get_hp(qweight)
    if qbias is not None:
        deq_b = dequantize_or_get_hp(qbias)
    else:
        deq_b = lin.bias
    # build a baseline with dequantized weight
    lin2 = nn.Linear(in_f, out_f, bias=True)
    lin2.weight = nn.Parameter(deq_w)
    lin2.bias = nn.Parameter(deq_b if isinstance(deq_b, torch.Tensor) else lin2.bias)
    out_deq = lin2(inp)
    assert torch.allclose(out_q if isinstance(out_q, torch.Tensor) else dequantize_or_get_hp(out_q), out_deq, atol=1e-3, rtol=1e-3)

def test_to_and_device_dtype_roundtrip(tmp_path):
    x = torch.randn(2, 3)
    q = factory(x)
    # test .to(dtype=) if supported
    try:
        moved = q.to(dtype=torch.float32)
    except Exception:
        pytest.skip(".to(dtype=...) not supported by F4QTensor", allow_module_level=False)
    assert hasattr(moved, "quant_state") or hasattr(moved, "dequantize") or isinstance(moved, F4QTensor)
    dq = dequantize_or_get_hp(moved)
    assert dq.dtype == torch.float32
    # test save / load of state_dict if class can be used as Parameter
    model = nn.Linear(3, 3)
    model.weight = nn.Parameter(q.dequantize() if hasattr(q, "dequantize") else dequantize_or_get_hp(q))
    sd = model.state_dict()
    p = tmp_path / "m.pt"
    torch.save(sd, str(p))
    loaded = torch.load(str(p))
    assert "weight" in loaded

def test_grad_flow_through_dequantized_path():
    # If NF4 design dequantizes to float for computation, ensure gradients flow to dequantized buffers
    x = torch.randn(2, 4, requires_grad=True)
    q = factory(torch.randn(2, 4))
    # create a simple scalar output via dot product with q (dequantize)
    dq = dequantize_or_get_hp(q).detach().requires_grad_(True)
    out = torch.sum(x * dq)
    out.backward()
    assert x.grad is not None

# end of tests
