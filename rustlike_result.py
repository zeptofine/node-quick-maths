from dataclasses import dataclass
from typing import Generic, Literal, NoReturn, TypeAlias, TypeVar, Union

T = TypeVar("T", covariant=True)
E = TypeVar("E", covariant=True)


@dataclass
class Ok(Generic[T]):
    __slots__ = ("_value",)

    _value: T

    def is_err(self) -> Literal[False]:
        return False

    def unwrap(self) -> T:
        return self._value

    def unwrap_err(self) -> NoReturn:
        raise Exception("Unwrapped an err on an Ok value")


@dataclass
class Err(Generic[E]):
    __slots__ = ("_value",)

    _value: E

    def is_err(self) -> Literal[True]:
        return True

    def unwrap(self) -> NoReturn:
        raise Exception("Unwrapped on an Err value")

    def unwrap_err(self) -> E:
        return self._value


Result: TypeAlias = Union[Ok[T], Err[E]]
