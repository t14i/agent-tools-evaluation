"""Common utilities and configurations for Temporal evaluation."""

import os
from dotenv import load_dotenv

load_dotenv()

TEMPORAL_ADDRESS = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
TEMPORAL_NAMESPACE = os.getenv("TEMPORAL_NAMESPACE", "default")
TASK_QUEUE = os.getenv("TEMPORAL_TASK_QUEUE", "evaluation-queue")


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(item: str, rating: str, note: str) -> None:
    """Print an evaluation result."""
    print(f"  {item}: {rating}")
    print(f"    → {note}\n")
