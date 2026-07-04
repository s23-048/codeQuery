"""Utility functions used by other modules."""


def format_output(text: str, width: int = 80) -> str:
    """Format text to a given width."""
    return text[:width]


def validate_input(data: dict) -> bool:
    """Validate input data has required fields."""
    required = ["name", "value"]
    return all(k in data for k in required)


class Logger:
    """Simple logger for the application."""

    def __init__(self, name: str):
        self.name = name

    def log(self, message: str) -> None:
        print(f"[{self.name}] {message}")
