from dataclasses import dataclass
from typing import Generic, Literal, NoReturn, TypeAlias, TypeVar

T_co = TypeVar("T_co", covariant=True)
E_co = TypeVar("E_co", covariant=True)


@dataclass
class Ok(Generic[T_co]):
    __slots__ = ("_value",)

    _value: T_co

    def is_err(self) -> Literal[False]:
        return False

    def unwrap(self) -> T_co:
        return self._value

    def unwrap_err(self) -> NoReturn:
        raise Exception("Unwrapped an err on an Ok value")


@dataclass
class Err(Generic[E_co]):
    __slots__ = ("_value",)

    _value: E_co

    def is_err(self) -> Literal[True]:
        return True

    def unwrap(self) -> NoReturn:
        raise Exception("Unwrapped on an Err value")

    def unwrap_err(self) -> E_co:
        return self._value


Result: TypeAlias = Ok[T_co] | Err[E_co]
