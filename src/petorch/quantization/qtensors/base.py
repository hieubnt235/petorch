import functools
from abc import abstractmethod, ABCMeta, ABC
from typing import Any, Self, TypeVar, Generic, Callable, TypeAlias, Sequence, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, SymInt, dtype, layout, strided, device

from petorch.utilities import fake_use
from petorch import logger


class QTensorConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class _WrapperState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    size: Sequence[int | SymInt]
    strides: Sequence[int | SymInt] | None = None
    storage_offset: int | SymInt | None = None
    memory_format: torch.memory_format | None = None
    dtype: dtype | None = None
    layout: layout = strided
    device: device | None = None
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


class QTensorState(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    quant_tensors: dict[str, Tensor] = Field(default_factory=dict)
    _wrapper_state: _WrapperState = Field(frozen=True)

    @model_validator(mode="after")
    def _check(self) -> Self:
        for v in self.quant_tensors.values():
            assert isinstance(v, Tensor)
        return self

    def _tensor_to(
        self, tensor_name: str, tensor: Tensor, *to_args: Any, **to_kwargs: Any
    ) -> Tensor:
        """
        Override this method
        to provide an appropriated ` to ` process applied to the states in the context of quantization
        Args:
            tensor_name:
            tensor:
            *to_args:
            **to_kwargs:

        Returns:
            New torch.Tensor instance
        """
        fake_use(self, tensor_name, str)
        return tensor.to(*to_args, **to_kwargs)

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
        quant_tensors_key = "quant_tensors"
        tensors: dict[str, Tensor] = {}
        for k, v in getattr(self, quant_tensors_key).items():
            assert isinstance(v, Tensor)
            tensors[k] = self._tensor_to(
                k, v, device, dtype, non_blocking, copy, memory_format=memory_format
            )
        states = self.model_dump(mode="python", exclude={quant_tensors_key})
        states[quant_tensors_key] = tensors
        new_state = self.__class__.model_validate(states)
        new_state._wrapper_state = self._wrapper_state.model_copy(
            update={"device": device, "dtype": dtype, "memory_format": memory_format},
            deep=True,
        )
        return new_state


QConfig_T = TypeVar("QConfig_T", bound=QTensorConfig)
QState_T = TypeVar("QState_T", bound=QTensorState)

AnyCallable: TypeAlias = Callable[..., Any]


class QTensorMeta(torch._C._TensorMeta, ABCMeta):
    pass


class QTensor(Tensor, ABC, Generic[QConfig_T, QState_T], metaclass=QTensorMeta):
    q_config_cls: type[QConfig_T]
    q_state_cls: type[QState_T]

    __TORCH_FUNCTIONS__: dict[AnyCallable, AnyCallable] = {}

    @classmethod
    @abstractmethod
    def _quantize_high_precision(
        cls, hp_tensor: Tensor, config: QConfig_T
    ) -> tuple[QConfig_T, QState_T]:
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
            getattr(cls, "q_config_cls"), QTensorConfig
        )
        assert hasattr(cls, "q_state_cls") and issubclass(
            getattr(cls, "q_state_cls"), QTensorState
        )

    @staticmethod
    def __new__(cls, *, _config: QConfig_T, _state: QTensorState) -> Self:
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

    def __init__(self, *, _config: QConfig_T, _state: QTensorState) -> None:
        assert isinstance(_config, self.q_config_cls)
        assert isinstance(_state, self.q_state_cls)

        super().__init__()
        self._config = _config
        self._state = _state

    @property
    def config(self) -> QConfig_T:
        return self._config

    @property
    def state(self) -> QState_T:
        return self._state

    @property
    def quant_tensors(self) -> dict[str, Tensor]:
        return self.state.quant_tensors

    @classmethod
    def from_high_precision(cls, hp_tensor: Tensor, config: QConfig_T) -> Self:
        config, state = cls._quantize_high_precision(hp_tensor, config)
        state._wrapper_state = _WrapperState.from_data(hp_tensor)
        return cls(_config=config, _state=state)

    def get_high_precision(self) -> Tensor:
        return self._dequantize_high_precision().to(
            device=self.device, dtype=self.dtype
        )

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
        kwargs = kwargs or {}
        if func in cls.__TORCH_FUNCTIONS__:
            return cls.__TORCH_FUNCTIONS__[func](*args, **kwargs)
        elif len(args) > 0 and isinstance(args[0], cls):
            hp_weight = args[0].get_high_precision()
            new_args = (hp_weight,)
            if len(args) > 1:
                new_args += args[1:]
        else:
            new_args = args
        return super().__torch_function__(func, types, new_args, kwargs)  # type: ignore

    def __setstate__(self, state: Any) -> None:
        pass  # TODO

    def __getstate__(self) -> None:
        pass  # TODO


@QTensor.implements_torch_function(torch.Tensor.to)
def function_to_dtype(*args: Any, **kwargs: Any) -> QTensor[QConfig_T, QState_T]:
    tensor = cast(QTensor[QConfig_T, QState_T], args[0])
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
        *args[1:], **kwargs
    )
    new_state = tensor.state.to(
        device, dtype, non_blocking, memory_format=convert_to_format
    )
    return tensor.__class__(_config=tensor.config, _state=new_state)
