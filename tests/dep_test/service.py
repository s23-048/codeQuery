"""Service that depends on utils and models."""

from utils import format_output, Logger
from models import User


class UserService:
    """Handles user operations."""

    def __init__(self):
        self.logger = Logger("UserService")

    def get_user_display(self, user: User) -> str:
        self.logger.log(f"Getting display for {user.name}")
        return format_output(f"{user.name} ({user.email})")


def create_user(name: str, email: str) -> User:
    """Factory function to create a User."""
    return User(name=name, email=email)
