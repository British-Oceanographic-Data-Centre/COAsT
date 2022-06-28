from dataclasses import dataclass
from numpy import number
from numbers import Number
from typing import Union, Optional


Numeric = Optional[Union[Number, number]]


@dataclass
class Coordinates2D:
    x: Numeric
    y: Numeric


@dataclass
class Coordinates3D(Coordinates2D):
    z: Numeric


@dataclass
class Coordinates4D(Coordinates3D):
    t: Numeric


Coordinates = Union[Coordinates2D, Coordinates3D, Coordinates4D]
