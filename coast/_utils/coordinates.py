from dataclasses import dataclass
from numpy import number, datetime64
from numbers import Number
from typing import Union, Optional


Numeric = Optional[Union[Number, number]]
Coordinate = Union[Numeric, datetime64]


@dataclass
class Coordinates2D:
    """Represent a point in one-to-two-dimensional space with optional X and Y coordinates."""

    x: Numeric
    y: Numeric


@dataclass
class Coordinates3D(Coordinates2D):
    """Represent a point in one-to-three-dimensional space with optional X, Y, and Z coordinates."""

    z: Numeric


@dataclass
class Coordinates4D(Coordinates3D):
    """Represent a point in one-to-four-dimensional spacetime with optional X, Y, Z, and T coordinates."""

    t: datetime64


Coordinates = Union[Coordinates2D, Coordinates3D, Coordinates4D]
