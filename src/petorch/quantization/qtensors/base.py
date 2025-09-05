import functools
from abc import abstractmethod, ABC, ABCMeta
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
)

import torch
from pydantic import BaseModel, ConfigDict, PrivateAttr
from torch import Tensor, SymInt, strided

from petorch import logger
from petorch.utilities import fake_use


class QConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, frozen=True
    )


class _WrapperState(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, frozen=True
    )
    size: Sequence[int | SymInt]
    strides: Sequence[int | SymInt] | None = None
    storage_offset: int | SymInt | None = None
    memory_format: torch.memory_format | None = None
    dtype: torch.dtype | None = None
    layout: torch.layout = strided
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


class QState(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, validate_default=True
    )

    _wrapper_state: _WrapperState | None = PrivateAttr(None)
    """This is private attribute, should not set it directly."""

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


QConfig_T = TypeVar("QConfig_T", bound=QConfig)
QState_T = TypeVar("QState_T", bound=QState)

AnyCallable: TypeAlias = Callable[..., Any]


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


class QTensor(Tensor, ABC, Generic[QConfig_T, QState_T], metaclass=QTensorMeta):
    q_config_cls: type[QConfig_T]
    q_state_cls: type[QState_T]

    __TORCH_FUNCTIONS__: dict[AnyCallable, AnyCallable] = {}
    _is_quantized: bool = True

    @classmethod
    @abstractmethod
    def _quantize_high_precision(cls, hp_tensor: Tensor, config: QConfig_T) -> QState_T:
        """

        Args:
            hp_tensor:
            config:

        Returns:

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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        assert hasattr(cls, "q_config_cls") and issubclass(
            getattr(cls, "q_config_cls"), QConfig
        )
        assert hasattr(cls, "q_state_cls") and issubclass(
            getattr(cls, "q_state_cls"), QState
        )

    @staticmethod
    def __new__(cls, *, _config: QConfig_T, _state: QState) -> Self:
        assert hasattr(_state, "_wrapper_state")
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

    def __init__(self, *, _config: QConfig_T, _state: QState) -> None:
        assert isinstance(_config, self.q_config_cls)
        assert isinstance(_state, self.q_state_cls)

        super().__init__()
        self._config = _config
        self._state = _state

    @property
    def config(self) -> QConfig_T:
        return self._config

    @property
    def quant_state(self) -> QState_T:
        return self._state

    @property
    def quant_tensors(self) -> dict[str, Tensor]:
        tensors = {}
        for name, ts in self.quant_state.iter_tensors():
            assert isinstance(ts, Tensor)
            tensors[name] = ts
        return tensors

    @classmethod
    def from_high_precision(cls, hp_tensor: Tensor, config: QConfig_T) -> Self:
        """

        Args:
            hp_tensor: A high-precision tensor.
            config: The instance of QTensorConfig.

        Returns:
            Instance of QTensor

        """
        assert torch.is_floating_point(hp_tensor)
        assert isinstance(config, getattr(cls, "q_config_cls"))

        state = cls._quantize_high_precision(hp_tensor, config)
        state._wrapper_state = _WrapperState.from_data(hp_tensor)
        return cls(_config=config, _state=state)

    def get_high_precision(self) -> Tensor:
        hpw = self._dequantize_high_precision().to(device=self.device, dtype=self.dtype)
        assert hpw.shape == self.shape
        return hpw

    @classmethod
    def implements_torch_function(
        cls, torch_function: AnyCallable
    ) -> Callable[[AnyCallable], AnyCallable]:
        if torch_function in cls.__TORCH_FUNCTIONS__:
            raise ValueError(f"Torch function already implemented: `{torch_function}`")

        def decorator(func: AnyCallable) -> AnyCallable:
            functools.update_wrapper(func, torch_function)
            cls.__TORCH_FUNCTIONS__[torch_function] = func
            logger.debug(f"{cls.__name__} implements torch function: `{func}`")
            return func

        return decorator

    @classmethod
    def __torch_function__(
        cls,
        func: AnyCallable,
        types: Sequence[type[Any]],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Intercepts Python-level torch API calls (like torch.add, torch.matmul, torch.mean) if the first argument is
        the subclass.
        Does not catch low-level operators like torch.ops or internal tensor ops in C++ kernels.
        """
        kwargs = kwargs or {}
        if func in cls.__TORCH_FUNCTIONS__:
            return cls.__TORCH_FUNCTIONS__[func](*args, **kwargs)
        elif len(args) > 0 and isinstance(args[0], cls):
            hp_tensor = args[0].get_high_precision()
            new_args = (hp_tensor,)
            if len(args) > 1:
                new_args += args[1:]
        else:
            new_args = args
        return super().__torch_function__(func, types, new_args, kwargs)  # type: ignore

    @classmethod
    def __torch_dispatch__(
        cls,
        func: AnyCallable,
        types: Sequence[type[Any]],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        """
        Called before __torch_function__ if the object is a torch.Tensor subclass registered with __torch_dispatch__.
        class MyTensor(torch.Tensor):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                print("Dispatch:", func.__name__)
                # Forward everything to base tensor
                return func(*args, **kwargs)

        x = torch.tensor([1, 2, 3]).as_subclass(MyTensor)
        y = x + x  # triggers __torch_dispatch__ first

        Args:
            func:
            types:
            args:
            kwargs:

        Returns:
            Any result returned by func should be either a regular tensor or wrapped back into your subclass,
            or else PyTorch may raise errors
        """
        # Default: forward all ops to the underlying tensor
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented
        kwargs = kwargs or {}
        results = func(*args, **kwargs)
        return results

    def __setstate__(self, state: Any) -> None:
        pass  # TODO

    def __getstate__(self) -> None:
        pass  # TODO


@QTensor.implements_torch_function(torch.Tensor.to)
def function_to_dtype(*args: Any, **kwargs: Any) -> QTensor[QConfig_T, QState_T]:
    qtensor = cast(QTensor[QConfig_T, QState_T], args[0])
    assert isinstance(qtensor, QTensor)
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
        *args[1:], **kwargs
    )
    new_state = qtensor.quant_state.to(
        device, dtype, non_blocking, memory_format=convert_to_format
    )
    return qtensor.__class__(_config=qtensor.config, _state=new_state)


if __name__ == "__main__":
    a = torch.tensor([[1, 2], [3, 4]])
    logger.info(a.shape)
    logger.info(a.__class__)
    logger.info(f"issubclass: {issubclass(QTensor,Tensor)}")  # True
    logger.info(f"isinstance: {isinstance(a,QTensor)}")  # False
