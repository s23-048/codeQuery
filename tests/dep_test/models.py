"""Data models for the application."""


class User:
    """Represents a user in the system."""

    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def __repr__(self) -> str:
        return f"User(name={self.name!r}, email={self.email!r})"


class Product:
    """Represents a product."""

    def __init__(self, title: str, price: float):
        self.title = title
        self.price = price
