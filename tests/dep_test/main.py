"""Main entry point — imports from service and models."""

from service import UserService, create_user
from models import User, Product
from utils import validate_input


def main():
    """Run the application."""
    service = UserService()
    user = create_user("Alice", "alice@example.com")
    print(service.get_user_display(user))

    data = {"name": "test", "value": 42}
    if validate_input(data):
        print("Input is valid!")
