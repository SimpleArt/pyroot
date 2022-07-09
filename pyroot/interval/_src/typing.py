from typing import Protocol, SupportsFloat, SupportsIndex, Union
from typing import runtime_checkable


@runtime_checkable
class SupportsRichComparison(Protocol):

    def __eq__(self, other: float) -> bool: ...
    def __ge__(self, other: float) -> bool: ...
    def __gt__(self, other: float) -> bool: ...
    def __le__(self, other: float) -> bool: ...
    def __lt__(self, other: float) -> bool: ...
    def __ne__(self, other: float) -> bool: ...


@runtime_checkable
class SupportsRichFloat(SupportsFloat, SupportsRichComparison, Protocol):
    pass


RealLike = Union[SupportsRichFloat, SupportsIndex]
