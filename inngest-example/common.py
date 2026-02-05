"""Common utilities and configurations for Inngest evaluation."""

import os
from dotenv import load_dotenv

load_dotenv()

# Inngest configuration
INNGEST_DEV_SERVER_URL = os.getenv("INNGEST_DEV_SERVER_URL", "http://127.0.0.1:8288")
INNGEST_APP_ID = os.getenv("INNGEST_APP_ID", "durable-eval")
INNGEST_SERVE_URL = os.getenv("INNGEST_SERVE_URL", "http://127.0.0.1:8000")

# OpenAI configuration (for AI integration tests)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(item: str, rating: str, note: str) -> None:
    """Print an evaluation result."""
    print(f"  {item}: {rating}")
    print(f"    -> {note}\n")
