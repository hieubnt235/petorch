import functools
from abc import abstractmethod, ABC, ABCMeta
from inspect import signature
from typing import (
    Any,
    Self,
    TypeVar,
    Generic,
    Callable,
    TypeAlias,
    Sequence,
    cast,
    ClassVar,
    Iterator,
    Literal,
    TYPE_CHECKING,
    overload,
)

import torch
from pydantic import BaseModel, ConfigDict, PrivateAttr
from torch import Tensor, SymInt
from torch.nn import functional as nnF
from torch.utils._python_dispatch import return_and_correct_aliasing

from petorch import logger
from petorch.utilities import fake_use

aten = torch.ops.aten
AnyCallable: TypeAlias = Callable[..., Any]
Device: TypeAlias = str | torch.device | int
Dtype: TypeAlias = str | torch.dtype


class _WrapperState(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, frozen=True
    )
    size: Sequence[int | SymInt]
    strides: Sequence[int | SymInt] | None = None
    storage_offset: int | SymInt | None = None
    memory_format: torch.memory_format | None = None
    dtype: torch.dtype | None = None
    layout: torch.layout = torch.strided
    device: torch.device | None = None
    pin_memory: bool = False
    requires_grad: bool = False

    @classmethod
    def from_data(cls, data: Tensor) -> Self:
        return cls(
            size=data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            # memory_format=None,
            dtype=data.dtype,
            layout=data.layout,
            device=data.device,
            # pin_memory=data.is_pinned(),
            requires_grad=data.requires_grad,
        )


class QConfig(BaseModel):
    """
    Config is the one to construct the QTensor. Two QTensors constructed by the same Tensor and Qconfig
    should have the same behavior. It's also the same part of QState.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, validate_default=True, validate_assignment=True
    )


QConfig_T = TypeVar("QConfig_T", bound=QConfig)


class QState(BaseModel, Generic[QConfig_T]):
    """
    State must contain all data information to reconstruct the high-precision tensor data,
    Two tensor data can be swaped by swaping the state.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, validate_default=True
    )

    config: QConfig_T

    _wrapper_state: _WrapperState | None = PrivateAttr(None)
    """This is private attribute, should not set it directly."""

    @property
    def wrapper_state(self):
        return self._wrapper_state

    @property
    def device(self) -> torch.device:
        return self._wrapper_state.device

    @property
    def dtype(self) -> torch.dtype:
        return self._wrapper_state.dtype

    @property
    def requires_grad(self) -> bool:
        return self._wrapper_state.requires_grad

    def iter_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for attr_name in self.__class__.model_fields.keys():
            t = getattr(self, attr_name)
            if isinstance(t, Tensor):
                yield attr_name, t

    # noinspection PyShadowingNames
    def to(
        self,
        device: str | torch.device | int | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> Self:
        """
        This method is called when QTensor.to(...) is invoked.
        Args:
            device:
            dtype:
            non_blocking:
            copy:
            memory_format:

        Returns:
            A new state, with new tensors and wrapper.

        """
        tensors: dict[str, Tensor] = {}
        for name, ts in self.iter_tensors():
            tensors[name] = self._tensor_to(
                name,
                ts,
                device,
                dtype,
                non_blocking,
                copy,
                memory_format=memory_format,
            )
        # Make a new state, hold all attributes except collected tensors
        states = self.model_dump(mode="python", exclude=set(tensors.keys()))
        states.update(tensors)
        new_state = self.__class__.model_validate(states)

        # Assign a new wrapper state
        new_state._wrapper_state = self._wrapper_state.model_copy(
            update={"device": device, "dtype": dtype, "memory_format": memory_format},
            deep=True,
        )
        return new_state

    def _tensor_to(
        self,
        tensor_name: str,
        tensor: Tensor,
        device: str | torch.device | int | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> Tensor:
        """
        Override this method
        to provide an appropriated ` to ` process applied to the states in the context of quantization
        Returns:
            New torch.Tensor instance
        """
        fake_use(self, tensor_name, str)
        return tensor.to(device, dtype, non_blocking, copy, memory_format=memory_format)


QState_T = TypeVar("QState_T", bound=QState)


class QTensorMeta(torch._C._TensorMeta, ABCMeta):
    # def __subclasscheck__(self, subclass: type)->bool:
    #     pass

    def __instancecheck__(self, instance: Any) -> bool:
        return (
            isinstance(instance, Tensor)
            and getattr(instance, "from_high_precision", None)
            and getattr(instance, "get_high_precision", None)
            and getattr(instance, "_is_quantized", None)
            and getattr(instance, "quant_state", None)
        )


ImplementFnOrOp: TypeAlias = Callable[
    [AnyCallable, Sequence[type[Any]], Sequence[Any], dict[str, Any]], Any
]
"""Ex: def _(func, types, args, kwargs)->Any: ..."""


class QTensor(Tensor, ABC, Generic[QState_T], metaclass=QTensorMeta):
    q_state_cls: type[QState_T]

    __TORCH_FUNCTIONS__: dict[AnyCallable, ImplementFnOrOp] = {}
    __TORCH_OPS__: dict[AnyCallable, ImplementFnOrOp] = {}
    _is_quantized: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        assert hasattr(cls, "q_state_cls") and issubclass(
            getattr(cls, "q_state_cls"), QState
        )

    @classmethod
    @abstractmethod
    def from_high_precision(cls, hp_tensor: Tensor, config: QConfig) -> Self:
        """
        Notes:
            Subclass must assign state._wrapper_state for the default __new__ method.

        assert torch.is_floating_point(hp_tensor)
        state = cls._quantize_high_precision(hp_tensor, config)
        state._wrapper_state = _WrapperState.from_data(hp_tensor)
        return cls(_state=state)


        Args:
            hp_tensor: A high-precision tensor.
            config:

        Returns:
            Instance of QTensor

        """
        ...

    @abstractmethod
    def _dequantize_high_precision(self) -> Tensor:
        """
        Get the original weight, should not convert to dtype or device in this method,
        they will be set to the same with the prequantized original weight.

        Returns:
            torch.Tensor instance
        """
        ...

    def get_high_precision(self) -> Tensor:
        hpw = self._dequantize_high_precision().to(device=self.device, dtype=self.dtype)
        assert hpw.shape == self.shape
        return hpw.requires_grad_(self.requires_grad)

    @staticmethod
    def __new__(cls, *, _state: QState_T) -> Self:
        """
        Create new method from state.
        Args:
            _state:
        """
        assert hasattr(_state, "_wrapper_state") and isinstance(
            _state._wrapper_state, _WrapperState
        )
        # noinspection PyProtectedMember
        wrapper = _state._wrapper_state

        return Tensor._make_wrapper_subclass(
            cls,
            size=wrapper.size,
            strides=wrapper.strides,
            storage_offset=wrapper.storage_offset,
            memory_format=wrapper.memory_format,
            dtype=wrapper.dtype,
            layout=wrapper.layout,
            device=wrapper.device,
            pin_memory=wrapper.pin_memory,
            requires_grad=wrapper.requires_grad,
        )

    def __init__(self, *, _state: QState_T) -> None:
        assert isinstance(_state, self.q_state_cls)
        super().__init__()
        self._state = _state

    @property
    def quant_state(self) -> QState_T:
        return self._state

    @property
    def config(self):
        return self.quant_state.config

    def iter_tensors(self) -> Iterator[tuple[str, Tensor]]:
        return self.quant_state.iter_tensors()

    @property
    def quant_tensors(self) -> dict[str, Tensor]:
        tensors = {}
        for name, ts in self.iter_tensors():
            assert isinstance(ts, Tensor)
            tensors[name] = ts
        return tensors

    def memory_footprint_in_bytes(self) -> int:
        n: int = 0
        for _, ts in self.quant_state.iter_tensors():
            assert isinstance(ts, Tensor)
            n += ts.numel() * ts.element_size()
        return n

    @classmethod
    def implements(
        cls,
        fns_or_ops: AnyCallable | Sequence[AnyCallable],
        *,
        mode: Literal["function", "dispatch"],
        force: bool = False,
    ) -> Callable[[ImplementFnOrOp], AnyCallable]:
        fns_or_ops = [fns_or_ops] if callable(fns_or_ops) else list(fns_or_ops)
        assert mode in ["function", "dispatch"]
        if mode == "dispatch":
            TORCH_FNS_OPS = cls.__TORCH_OPS__
            _n = "op"
        else:
            TORCH_FNS_OPS = cls.__TORCH_FUNCTIONS__
            _n = "fn"

        def decorator(func: ImplementFnOrOp) -> ImplementFnOrOp:
            assert (
                len(signature(func).parameters) == 4
            ), f"Op or function must have signature (func, types, args, kwargs)"
            for fn_op in fns_or_ops:
                if fn_op in TORCH_FNS_OPS and not force:
                    raise ValueError(
                        f"Torch function or op already implemented: `{fns_or_ops}`. Explicitly set `force=True` to override."
                    )
                TORCH_FNS_OPS[fn_op] = func
                logger.debug(
                    f"`{cls.__name__}` implements torch {_n} `{fn_op}` with `{func}`"
                )
                functools.update_wrapper(func, fn_op)
            return func

        return decorator

    def __repr__(self, *, __wrapper: bool = False) -> str:
        # This override is importance, if not, maximum recursion error will be raised when you print out in the __torch_function__
        # because the default repr will call backend attribute, such as self.dtype, which call __torch_function__ again, and recurse forever.
        # Use the __wrapper=True for not directly call self.
        obj = self.quant_state if __wrapper else self

        return (
            f"{self.__class__.__name__}("
            f"quant_state: {self.quant_state.__repr__()},"
            f"device={obj.device},"
            f"dtype={obj.dtype},"
            f"requires_grad={obj.requires_grad},"
            f"memory_footprint: {self.memory_footprint_in_bytes()} bytes)"
        )

    def __str__(self, *, __wrapper: bool = False) -> str:
        obj = self.quant_state if __wrapper else self
        return (
            f"{self.__class__.__name__}("
            f"device={obj.device},"
            f"dtype={obj.dtype},"
            f"requires_grad={obj.requires_grad},"
            f"memory_footprint: {self.memory_footprint_in_bytes()} bytes)"
        )

    def __setstate__(self, state: Any) -> None:
        pass  # TODO

    def __getstate__(self) -> None:
        pass  # TODO

    @classmethod
    def __torch_function__(
        cls,
        func: AnyCallable,
        types: Sequence[type[Any]],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        logger.debug(f"{getattr(func,"__self__", None) or func},{types=}")
        assert any(issubclass(cls, t) for t in types)  # At least one QTensor

        kwargs = kwargs or {}
        if func in cls.__TORCH_FUNCTIONS__:
            return cls.__TORCH_FUNCTIONS__[func](func, types, args, kwargs)

        # Disable torch_function because we don't want the wrapping behavior of the super() impl.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(
        cls,
        func: AnyCallable,
        types: Sequence[type[Any]],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        logger.debug(f"{func=}, {types=}, {args=}, {kwargs=}")
        assert any(issubclass(cls, t) for t in types)  # At least one QTensor

        kwargs = kwargs or {}
        if func in cls.__TORCH_OPS__:
            return cls.__TORCH_OPS__[func](func, types, args, kwargs)
        else:
            raise NotImplementedError(
                f"{func=} | {types=} | {[type(arg) for arg in args]} | {kwargs}"
            )

    if TYPE_CHECKING:

        @overload
        def to(
            self,
            dtype: Device | None = None,
            non_blocking: bool = False,
            copy: bool = False,
            *,
            memory_format: torch.memory_format | None = None,
        ) -> Self: ...

        @overload
        def to(
            self,
            device: Device | None = None,
            dtype: Dtype | None = None,
            non_blocking: bool = False,
            copy: bool = False,
            *,
            memory_format: torch.memory_format | None = None,
        ) -> Self: ...

        @overload
        def to(
            self,
            other: Tensor,
            non_blocking: bool = False,
            copy: bool = False,
            *,
            memory_format: torch.memory_format | None = None,
        ) -> Self: ...
        def to(self, *args, **kwargs) -> Self: ...
        def cuda(
            self,
            device: Device | None = None,
            non_blocking: bool = False,
            memory_format: torch.memory_format = torch.preserve_format,
        ) -> Self: ...
        def cpu(
            self,
            memory_format=torch.preserve_format,
        ) -> Self: ...
        def requires_grad_(self, mode=True) -> Self: ...
        def detach(self) -> Self: ...
        def clone(self, *, memory_format=None) -> Self: ...


TorchAutogradFunction = torch.autograd.Function


class QLinear(TorchAutogradFunction):
    """
    Three quantize cases: weight only, bias only, weight and bias
    """

    @staticmethod
    def forward(
        input_tensor: Tensor,
        weight: Tensor,
        bias: None | Tensor = None,
    ) -> Any:
        hp_weight = (
            weight.get_high_precision() if isinstance(weight, QTensor) else weight
        )
        hp_bias = bias.get_high_precision() if isinstance(bias, QTensor) else bias
        return nnF.linear(input_tensor, hp_weight, hp_bias)

    @staticmethod
    def setup_context(
        ctx: TorchAutogradFunction,
        inputs: tuple[Tensor, Tensor, None | Tensor],
        output: Tensor,
    ) -> Any:
        # No need to save bias.
        ctx.save_for_backward(*inputs[:-1])

    @staticmethod
    def backward(ctx: TorchAutogradFunction, *grad_outputs: Tensor) -> Any:
        input_tensor, weight = cast(tuple[Tensor, Tensor], ctx.saved_tensors)
        weight = weight.get_high_precision() if isinstance(weight, QTensor) else weight

        rq_ip, rq_w, rq_b = cast(tuple[bool, bool, bool], ctx.needs_input_grad)
        grad_out = grad_outputs[0]
        grad_ip = grad_w = grad_b = None

        # Note about shapes: grad_out=(N, out_c), weight=(out_c, in_c), input_tensors=(N, in_c)
        # DY/DX must have the same shape of X.
        if rq_ip:
            grad_ip = grad_out @ weight
        if rq_w:
            grad_w = torch.matmul(
                grad_out.T,
                input_tensor,
            )
        if rq_b:
            # Gradient wrt bias = 1. Just sum the batch.
            grad_b = grad_out.sum(dim=0)
        return grad_ip, grad_w, grad_b

    if TYPE_CHECKING:

        @classmethod
        def apply(
            cls,
            input_tensor: Tensor,
            weight: Tensor,
            bias: None | Tensor = None,
        ) -> Tensor: ...


class QConvNd(TorchAutogradFunction):

    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> Any:
        pass

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        pass

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass


class QEmbedding(TorchAutogradFunction):

    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> Any:
        pass

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        pass

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass


##############
# Implements #
##############
implements = QTensor.implements


@implements(nnF.linear, mode="function")
def func_linear(
    func, types, args: tuple[Tensor, Tensor, None | Tensor], kwargs: Any
) -> Tensor:
    fake_use(types, kwargs)
    assert func == nnF.linear
    assert len(args) == 3 and not kwargs
    input_tensor, weight, bias = args
    assert isinstance(input_tensor, Tensor)
    assert isinstance(weight, QTensor) or isinstance(bias, QTensor)

    return QLinear.apply(input_tensor, weight, bias)


@implements(aten.add.Tensor, mode="dispatch")
def op_add(func, types, args: tuple[Tensor, Tensor], kwargs) -> QTensor:
    fake_use(types)
    assert len(args) == 2
    tensors = [t.get_high_precision() if isinstance(t, QTensor) else t for t in args]

    out = torch.add(*tensors, **kwargs)
    return return_and_correct_aliasing(func, args, kwargs, out)


@implements(aten.add_.Tensor, mode="dispatch")
def op_add_(func, types, args: tuple[Tensor, Tensor], kwargs: Any) -> QTensor:
    fake_use(types)
    assert len(args) == 2
    tensors = [t.get_high_precision() if isinstance(t, QTensor) else t for t in args]

    if isinstance(args[0], QTensor):
        qt = cast(QTensor, args[0])
        new_value = QTensor.from_high_precision(
            torch.add(*tensors, **kwargs), qt.config
        )
        qt._state = new_value.quant_state
    else:
        cast(Tensor, args[0]).add_(tensors[1])

    return return_and_correct_aliasing(func, args, kwargs, args[0])


@implements([aten._to_copy.default, aten.to.dtype, aten.to.device], mode="dispatch")
def op_to(
    func, types, args: tuple[QTensor[QState_T]], kwargs: Any
) -> QTensor[QState_T]:
    """
    cuda, cpu, to,... will call this op
    """
    fake_use(func, types)
    allowed_k = ["device", "dtype", "non_blocking", "copy", "memory_format"]
    kwargs = {k: v for k, v in kwargs.items() if k in allowed_k}

    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(**kwargs)

    new_state = args[0].quant_state.to(
        device, dtype, non_blocking, memory_format=convert_to_format
    )
    return args[0].__class__(_state=new_state)


def _apply_to_state(
    fn: Callable[[Tensor], Tensor], q_tensor: QTensor[QState_T]
) -> QTensor[QState_T]:
    """
    Apply `fn` to all data tensors to create a new one. Then return the same state with a new tensor.
    Note that the conflict of tensors (such as the `fn` return view, or apply inplace modifying will
    cause unbehavior.
    Args:
        fn:

    Returns:
        QTensor subclass instance.
    """

    tensors: dict[str, Tensor] = {}
    for name, ts in q_tensor.quant_state.iter_tensors():
        new_ts = fn(ts)
        assert type(new_ts) is Tensor
        tensors[name] = new_ts

    # Make a new state, hold all attributes except collected tensors
    states = q_tensor.quant_state.model_dump(mode="python", exclude=set(tensors.keys()))
    states.update(tensors)
    new_state = q_tensor.quant_state.model_validate(states)

    # Assign a new wrapper state
    new_state._wrapper_state = q_tensor.quant_state._wrapper_state.model_copy()
    return q_tensor.__class__(_state=new_state)


@implements([aten.detach.default, aten.clone.default], mode="dispatch")
def op_detach_clone(
    func, types, args: tuple[QTensor[QState_T]], kwargs
) -> QTensor[QState_T]:
    assert not kwargs and issubclass(types[0], QTensor)
    assert isinstance(args[0], QTensor)
    return _apply_to_state(func, args[0])


@implements(aten.t.default, mode="dispatch")
def op_t(func, types, args: tuple[QTensor[QState_T]], kwargs) -> QTensor[QState_T]:
    assert not kwargs and issubclass(types[0], QTensor)
    assert isinstance((qt := args[0]), QTensor)
    return qt.from_high_precision(qt.get_high_precision().t(), qt.config)


@implements(aten.copy_.default, mode="dispatch")
def op_copy_(
    func, types, args: tuple[Tensor, Tensor], kwargs: dict[str, Any]
) -> Tensor:
    fake_use(func, types)
    dest, src = args
    assert dest.shape == src.shape
    non_blocking = kwargs.pop("non_blocking", False)
    # Covert to hp, because maybe it does not have the same config as the dest.
    src = src.get_high_precision() if isinstance(src, QTensor) else src

    if isinstance(dest, QTensor):
        dest._state = (
            dest.from_high_precision(src, dest.config)
            .to(device=dest.device, dtype=dest.dtype, non_blocking=non_blocking)
            .quant_state
        )
    else:
        dest.copy_(src, non_blocking=non_blocking)

    return dest


aten = torch.ops.aten
common_tensor_ops = [
    torch.Tensor.contiguous,
    aten.alias.default,
    aten.contiguous.default,
    torch.bmm,
    aten.slice.Tensor,
    aten.cat.default,
    aten.transpose.int,
    aten.view.default,
    aten.squeeze.dim,
    aten.copy_.default,
    aten.addmm.default,
    aten.mm.default,
    aten.permute.default,
]


# @QTensor.implements(torch.Tensor.to, mode="function")
# def function_to(func, types, args: Any, kwargs: Any) -> QTensor[QConfig_T, QState_T]:
#     assert func == torch.Tensor.to
#     logger.debug(f"{func=}, {types=}, {kwargs=}")
#     qtensor = cast(QTensor[QConfig_T, QState_T], args[0])
#     assert isinstance(qtensor, QTensor)
#     device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
#         *args[1:], **kwargs
#     )
#     new_state = qtensor.quant_state.to(
#         device, dtype, non_blocking, memory_format=convert_to_format
#     )
#     return qtensor.__class__(_config=qtensor.config, _state=new_state)
