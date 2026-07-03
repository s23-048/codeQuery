"""Test file for Python parsing — has a mix of constructs."""

import os
from pathlib import Path


def standalone_function(x: int, y: int) -> int:
    """A plain top-level function."""
    return x + y


class Calculator:
    """A simple calculator class with methods."""

    def __init__(self, precision: int = 2):
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        return round(a + b, self.precision)

    def multiply(self, a: float, b: float) -> float:
        return round(a * b, self.precision)


def another_function():
    """Another top-level function after the class."""
    pass


# Some module-level code
CONSTANT = 42
