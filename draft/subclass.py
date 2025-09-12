import time
from typing import cast
import wrapt

import torch
from torch import nn
from triton.language import bfloat16

aten = torch.ops.aten
import torch.utils._pytree as pytree


@wrapt.decorator
def print_torch_func(wrapped, instance, args, kwargs):
    print(f"{wrapped=}, {instance=}, {args=}, {kwargs} ")
    return wrapped(*args, **kwargs)


# torch.Tensor.__torch_function__ = print_torch_func(torch.Tensor.__torch_function__)


class MyTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, a: torch.Tensor, name="noname"):
        tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            a.shape,
            strides=a.stride(),
            storage_offset=a.storage_offset(),
            dtype=a.dtype,
            device=a.device,
            requires_grad=a.requires_grad,
        )
        return tensor

    def __init__(self, a: torch.Tensor, name="noname"):
        self.a = a
        self.tensor_name = name

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        def repr_arg(t):
            if isinstance(t, cls):
                return t.tensor_name
            else:
                return type(t)

        print(f"FUNC:{"-"*100}\n"
              f" {func=}, {types=} | {[repr_arg(a) for a in args]} | { {k:v for k,v in kwargs.items()} }\n"
              f"{"-"*100}")

        # if not all(issubclass(cls, t) for t in types):
        #     return NotImplemented

        with torch._C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)
            return ret

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}

        def repr_arg(t):
            if isinstance(t, cls):
                return t.tensor_name
            else:
                return type(t)

        print(
            f"OPS:{"-"*100}\n"
            f" {func=}, {types=} | {[repr_arg(a) for a in args]} | { {k:v for k,v in kwargs.items()} }\n"
            f"{"-"*100}"
        )

        print("~" * 100)

        args_a = pytree.tree_map_only(MyTensor, lambda x: x.a, args)
        kwargs_a = pytree.tree_map_only(MyTensor, lambda x: x.a, kwargs)
        out_a = func(*args_a, **kwargs_a)
        if func in [aten.detach.default, aten.t.default]:
            return cls(out_a, name=args[0].tensor_name)

        out_a_flat, spec = pytree.tree_flatten(out_a)
        out_flat = [
            cls(o_a) if isinstance(o_a, torch.Tensor) else o_a for o_a in out_a_flat
        ]
        r = pytree.tree_unflatten(out_flat, spec)
        print(f"return {type(r)}")
        return r

    def __repr__(self, *, tensor_contents=None):
        return f"{self.a}"


# lm = nn.Linear(64, 32, bias=False)
# q_ts = MyTensor(lm.weight, "weight")
# q_ts2 = q_ts.detach().clone()

a = MyTensor(torch.randn([2,3],device="cuda"), name="a")
b = MyTensor(torch.randn([2,3],device="cpu"), name="a")
b.copy_(a)
c = a.to(device = torch.device("cpu"), dtype=torch.bfloat16)

# a.t()
# a.T
# a.clone()
# a.to(b)
# b = MyTensor(a.detach().clone().a, name="b")
# c = a.detach().clone().cpu()

# print("\nTensor.add==============================\n")
# print("a+b")
# a + b
# print("torch.add(a,b)")
# torch.add(a,b)
#
# print("a.add(b)")
# a.add(b)
# print("a.add_(b)")
# a.add_(b)
# print("a+c")
# a+c.cuda()
# print("c+a")
# c+a.cpu()
# c = c.to("cuda")
# a.add_(c)
# c.add_(a)
# a.add(c)
# c.add(a)
# d = a.add(b)
# a.add_(b)
# torch.add(a,b)

# bias_ts = MyTensor(torch.randn([1, 32]), "bias")
#
# lm.weight = nn.Parameter(q_ts)
# lm.bias = nn.Parameter(bias_ts)
#
# input = torch.randn([2, 64])
#
# print("=" * 100)
# output = lm(input)
# print("=" * 100)
# print(output.tensor_name)
# print(type(output))
#
# class DummyTensor(torch.Tensor):
#     pass
#
# t = DummyTensor([1,2,3])
# # t = torch.tensor([1,2,3])
# t2 = t.requires_grad_(False)
# print(t2)

# t = MyTensor(torch.randn([3, 5], requires_grad=True))
# x = MyTensor(torch.rand([5,3]))
#
# output = t@x
# print(output.grad_fn)
# output1 = output.sum()
# print(output.grad_fn, output1.grad_fn)
# print("grad: ",t.grad, t.a.grad, )
# output1.backward()
# print(output.grad_fn)
#
# print("grad: ", t.grad)
# print(id (t.grad),id (t.a.grad))

# print(t)
# s = t+5
# print(s.device)
# print(s)
# print(t.data)
# print(s.data)
# print(s.detach())
