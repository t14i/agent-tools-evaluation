"""
15_determinism_replay.py - Determinism and Replay Verification (DR-01, DR-04)

Purpose: Verify replay (reproducible execution) and idempotency
- Fixed LLM output with seed, temperature=0
- Mocking external I/O for reproducibility
- commit_key equivalent implementation
- Duplicate execution detection and prevention

This is a FAIL-CLOSE requirement: operations must be reproducible and idempotent.
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Optional, Type
from pathlib import Path

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crewai.flow.flow import Flow, listen, start


# =============================================================================
# Constants
# =============================================================================

DB_PATH = Path("./db")
DB_PATH.mkdir(exist_ok=True)

REPLAY_LOG_PATH = DB_PATH / "replay_log.jsonl"
IDEMPOTENCY_STORE_PATH = DB_PATH / "idempotency_store.json"


# =============================================================================
# Idempotency Key Manager
# =============================================================================

class IdempotencyKeyManager:
    """
    Manages idempotency keys to prevent duplicate execution (DR-04).

    Each operation is assigned a unique key based on its parameters.
    If the same key is seen again, the previous result is returned
    instead of re-executing.
    """

    def __init__(self, store_path: Path = IDEMPOTENCY_STORE_PATH):
        self.store_path = store_path
        self.store = self._load_store()

    def _load_store(self) -> dict:
        """Load the idempotency store from disk."""
        if self.store_path.exists():
            with open(self.store_path, "r") as f:
                return json.load(f)
        return {}

    def _save_store(self):
        """Save the idempotency store to disk."""
        with open(self.store_path, "w") as f:
            json.dump(self.store, f, indent=2, default=str)

    def generate_key(self, operation: str, params: dict) -> str:
        """
        Generate an idempotency key from operation and parameters.

        Uses SHA256 hash of operation + sorted params JSON.
        """
        data = json.dumps({"operation": operation, "params": params}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def check_and_record(self, key: str, result: Any) -> tuple[bool, Optional[Any]]:
        """
        Check if key exists. If so, return cached result.
        Otherwise, record the new result.

        Returns:
            (is_duplicate, cached_result_or_none)
        """
        if key in self.store:
            cached = self.store[key]
            print(f"  [Idempotency] Duplicate detected! Key: {key}")
            print(f"  [Idempotency] Returning cached result from {cached['timestamp']}")
            return True, cached["result"]

        # Record new execution
        self.store[key] = {
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }
        self._save_store()
        return False, None

    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result for a key if it exists."""
        if key in self.store:
            return self.store[key]["result"]
        return None

    def clear(self):
        """Clear all stored keys."""
        self.store = {}
        self._save_store()


# =============================================================================
# Replay Logger
# =============================================================================

class ReplayLogger:
    """
    Logs all operations for replay capability (DR-01).

    Each execution is logged with:
    - Timestamp
    - Input parameters
    - Output/result
    - Random seed (if applicable)
    """

    def __init__(self, log_path: Path = REPLAY_LOG_PATH):
        self.log_path = log_path

    def log_execution(self, operation: str, inputs: dict, output: Any,
                      seed: Optional[int] = None, metadata: Optional[dict] = None):
        """Log an execution for potential replay."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "inputs": inputs,
            "output": output,
            "seed": seed,
            "metadata": metadata or {},
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        return entry

    def get_all_executions(self) -> list[dict]:
        """Read all logged executions."""
        executions = []
        if self.log_path.exists():
            with open(self.log_path, "r") as f:
                for line in f:
                    if line.strip():
                        executions.append(json.loads(line))
        return executions

    def replay_execution(self, execution: dict) -> dict:
        """
        Replay a logged execution.

        In a real implementation, this would:
        1. Set the same random seed
        2. Mock external dependencies with logged values
        3. Re-execute the operation
        """
        print(f"\n[Replay] Replaying execution from {execution['timestamp']}")
        print(f"[Replay] Operation: {execution['operation']}")
        print(f"[Replay] Inputs: {execution['inputs']}")
        print(f"[Replay] Expected output: {execution['output']}")

        return {
            "original": execution,
            "replayed": True,
            "status": "Replay simulation successful",
        }

    def clear(self):
        """Clear the replay log."""
        if self.log_path.exists():
            self.log_path.unlink()


# =============================================================================
# Input Snapshot Manager
# =============================================================================

class InputSnapshotManager:
    """
    Captures and stores input snapshots for reproducibility.

    This allows replaying with the exact same inputs even if
    external sources have changed.
    """

    def __init__(self, snapshot_dir: Path = DB_PATH / "snapshots"):
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(exist_ok=True)

    def capture_snapshot(self, snapshot_id: str, data: dict) -> str:
        """Capture an input snapshot."""
        path = self.snapshot_dir / f"{snapshot_id}.json"
        with open(path, "w") as f:
            json.dump({
                "id": snapshot_id,
                "captured_at": datetime.now().isoformat(),
                "data": data,
            }, f, indent=2, default=str)
        return str(path)

    def load_snapshot(self, snapshot_id: str) -> Optional[dict]:
        """Load a previously captured snapshot."""
        path = self.snapshot_dir / f"{snapshot_id}.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None


# =============================================================================
# Idempotent Tool Implementation
# =============================================================================

class IdempotentInput(BaseModel):
    """Input schema for idempotent operations."""
    operation_id: str = Field(..., description="Unique identifier for this operation")
    data: str = Field(..., description="Data to process")


class IdempotentProcessorTool(BaseTool):
    """
    A tool that demonstrates idempotent execution.

    If the same operation_id + data combination is seen twice,
    returns the cached result instead of re-processing.
    """

    name: str = "Idempotent Processor"
    description: str = """Process data with idempotency guarantee.
    Same operation_id + data will return cached result."""
    args_schema: Type[BaseModel] = IdempotentInput

    idempotency_manager: IdempotencyKeyManager = None
    replay_logger: ReplayLogger = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.idempotency_manager = IdempotencyKeyManager()
        self.replay_logger = ReplayLogger()

    def _run(self, operation_id: str, data: str) -> str:
        """Process data with idempotency check."""
        # Generate idempotency key
        params = {"operation_id": operation_id, "data": data}
        key = self.idempotency_manager.generate_key("process", params)

        print(f"  [IdempotentProcessor] Operation: {operation_id}")
        print(f"  [IdempotentProcessor] Idempotency key: {key}")

        # Check for duplicate
        is_duplicate, cached = self.idempotency_manager.check_and_record(
            key, None  # We'll update with actual result
        )

        if is_duplicate and cached is not None:
            return f"[CACHED] {cached}"

        # Process the data (simulated)
        result = f"Processed '{data}' for operation {operation_id} at {datetime.now().isoformat()}"

        # Update the store with actual result
        self.idempotency_manager.store[key]["result"] = result
        self.idempotency_manager._save_store()

        # Log for replay
        self.replay_logger.log_execution(
            operation="process",
            inputs=params,
            output=result,
        )

        return result


# =============================================================================
# Deterministic Flow State
# =============================================================================

class DeterministicState(BaseModel):
    """State for deterministic workflow."""

    execution_id: str = ""
    input_snapshot_id: str = ""
    processed_ids: list[str] = []  # Track processed items for dedup
    results: list[dict] = []
    seed: int = 42  # Fixed seed for reproducibility
    is_replay: bool = False


# =============================================================================
# Deterministic Workflow
# =============================================================================

class DeterministicWorkflow(Flow[DeterministicState]):
    """
    Workflow demonstrating deterministic execution.

    Key features:
    - Fixed random seed for reproducibility
    - Input snapshotting
    - Duplicate detection via processed_ids
    - Execution logging for replay
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.replay_logger = ReplayLogger()
        self.snapshot_manager = InputSnapshotManager()

    @start()
    def initialize(self):
        """Initialize workflow with deterministic settings."""
        print("\n[Initialize] Setting up deterministic execution...")

        if not self.state.execution_id:
            self.state.execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Set fixed seed for reproducibility
        import random
        random.seed(self.state.seed)

        print(f"[Initialize] Execution ID: {self.state.execution_id}")
        print(f"[Initialize] Random seed: {self.state.seed}")
        print(f"[Initialize] Is replay: {self.state.is_replay}")

    @listen(initialize)
    def capture_inputs(self):
        """Capture input snapshot for replay capability."""
        print("\n[Capture] Capturing input snapshot...")

        # Simulate input data
        inputs = {
            "items": ["item_1", "item_2", "item_3"],
            "config": {"mode": "test", "verbose": True},
        }

        if not self.state.is_replay:
            self.state.input_snapshot_id = f"snapshot_{self.state.execution_id}"
            self.snapshot_manager.capture_snapshot(
                self.state.input_snapshot_id,
                inputs
            )
            print(f"[Capture] Snapshot saved: {self.state.input_snapshot_id}")
        else:
            # Load from snapshot for replay
            snapshot = self.snapshot_manager.load_snapshot(self.state.input_snapshot_id)
            if snapshot:
                inputs = snapshot["data"]
                print(f"[Capture] Loaded snapshot from {snapshot['captured_at']}")

        return inputs

    @listen(capture_inputs)
    def process_items(self, inputs: dict):
        """Process items with deduplication."""
        print("\n[Process] Processing items with deduplication...")

        items = inputs.get("items", [])

        for item in items:
            # Check if already processed (deduplication)
            if item in self.state.processed_ids:
                print(f"[Process] Skipping duplicate: {item}")
                continue

            # Process the item
            result = {
                "item": item,
                "processed_at": datetime.now().isoformat(),
                "execution_id": self.state.execution_id,
            }

            self.state.processed_ids.append(item)
            self.state.results.append(result)
            print(f"[Process] Processed: {item}")

        # Log for replay
        self.replay_logger.log_execution(
            operation="process_items",
            inputs=inputs,
            output=self.state.results,
            seed=self.state.seed,
            metadata={"execution_id": self.state.execution_id},
        )

    @listen(process_items)
    def finalize(self):
        """Finalize and report."""
        print("\n[Finalize] Generating report...")

        return f"""
Deterministic Execution Report
==============================
Execution ID: {self.state.execution_id}
Snapshot ID: {self.state.input_snapshot_id}
Random Seed: {self.state.seed}
Is Replay: {self.state.is_replay}
Items Processed: {len(self.state.results)}
Processed IDs: {self.state.processed_ids}
"""


# =============================================================================
# Demonstrations
# =============================================================================

def demo_idempotency():
    """Demonstrate idempotency key mechanism."""
    print("=" * 60)
    print("Demo 1: Idempotency Key Mechanism (DR-04)")
    print("=" * 60)

    manager = IdempotencyKeyManager()
    manager.clear()  # Start fresh

    # First execution
    print("\n--- First Execution ---")
    key1 = manager.generate_key("process", {"id": "123", "data": "test"})
    print(f"Generated key: {key1}")
    is_dup, cached = manager.check_and_record(key1, "Result A")
    print(f"Is duplicate: {is_dup}, Cached: {cached}")

    # Second execution with same params (should be duplicate)
    print("\n--- Second Execution (same params) ---")
    key2 = manager.generate_key("process", {"id": "123", "data": "test"})
    print(f"Generated key: {key2}")
    is_dup, cached = manager.check_and_record(key2, "Result B")
    print(f"Is duplicate: {is_dup}, Cached: {cached}")

    # Third execution with different params
    print("\n--- Third Execution (different params) ---")
    key3 = manager.generate_key("process", {"id": "456", "data": "test"})
    print(f"Generated key: {key3}")
    is_dup, cached = manager.check_and_record(key3, "Result C")
    print(f"Is duplicate: {is_dup}, Cached: {cached}")


def demo_replay_logging():
    """Demonstrate replay logging mechanism."""
    print("\n" + "=" * 60)
    print("Demo 2: Replay Logging (DR-01)")
    print("=" * 60)

    logger = ReplayLogger()
    logger.clear()  # Start fresh

    # Log some executions
    print("\n--- Logging executions ---")
    logger.log_execution(
        operation="analyze",
        inputs={"text": "Hello world"},
        output={"sentiment": "positive"},
        seed=42,
    )
    logger.log_execution(
        operation="summarize",
        inputs={"document": "Long document..."},
        output={"summary": "Short summary"},
        seed=42,
    )

    # Read back executions
    print("\n--- Reading execution log ---")
    executions = logger.get_all_executions()
    for i, exe in enumerate(executions):
        print(f"\nExecution {i + 1}:")
        print(f"  Operation: {exe['operation']}")
        print(f"  Timestamp: {exe['timestamp']}")
        print(f"  Seed: {exe['seed']}")

    # Replay an execution
    if executions:
        print("\n--- Replaying first execution ---")
        logger.replay_execution(executions[0])


def demo_deterministic_workflow():
    """Demonstrate deterministic workflow execution."""
    print("\n" + "=" * 60)
    print("Demo 3: Deterministic Workflow")
    print("=" * 60)

    # First execution
    print("\n--- First Execution ---")
    flow1 = DeterministicWorkflow()
    result1 = flow1.kickoff()
    print(result1)

    # Replay execution with same seed
    print("\n--- Replay Execution ---")
    flow2 = DeterministicWorkflow()
    flow2.state.is_replay = True
    flow2.state.seed = 42  # Same seed
    flow2.state.input_snapshot_id = flow1.state.input_snapshot_id
    result2 = flow2.kickoff()
    print(result2)


def demo_idempotent_tool():
    """Demonstrate idempotent tool with CrewAI agent."""
    print("\n" + "=" * 60)
    print("Demo 4: Idempotent Tool with Agent")
    print("=" * 60)

    tool = IdempotentProcessorTool()
    tool.idempotency_manager.clear()

    # First call
    print("\n--- First Tool Call ---")
    result1 = tool._run(operation_id="op_001", data="test data")
    print(f"Result: {result1}")

    # Second call with same params (should return cached)
    print("\n--- Second Tool Call (same params) ---")
    result2 = tool._run(operation_id="op_001", data="test data")
    print(f"Result: {result2}")

    # Third call with different params
    print("\n--- Third Tool Call (different params) ---")
    result3 = tool._run(operation_id="op_002", data="different data")
    print(f"Result: {result3}")


def demo_llm_determinism_config():
    """Show how to configure LLM for determinism."""
    print("\n" + "=" * 60)
    print("Demo 5: LLM Determinism Configuration")
    print("=" * 60)

    print("""
LLM Determinism Settings:
=========================

For reproducible LLM outputs, configure:

1. Temperature = 0 (or very low)
   - Reduces randomness in token selection
   - Same input should produce same output

2. Seed parameter (if supported by provider)
   - OpenAI: seed parameter in API call
   - Some providers don't support this

3. Top-p = 1.0
   - Use full probability distribution

CrewAI Configuration Example:
-----------------------------
from crewai import LLM

llm = LLM(
    model="gpt-4o",
    temperature=0,
    # seed=42,  # If supported
)

agent = Agent(
    role="...",
    goal="...",
    backstory="...",
    llm=llm,
)

Important Notes:
- Even with temperature=0, outputs may vary slightly
- External tool calls add non-determinism
- For true reproducibility, mock all external I/O
- Log all LLM calls for replay capability
""")


def main():
    print("=" * 60)
    print("Determinism & Replay Verification (DR-01, DR-04)")
    print("=" * 60)
    print("""
This script verifies replay (reproducible execution) and idempotency.

Verification Items:
1. DR-01: Replay - Fixed seed, mocked I/O, reproducible execution
2. DR-04: Idempotency - Exactly-once execution, duplicate detection

Key Concepts:
- Idempotency Key: Hash of operation + params to detect duplicates
- Replay Log: Complete record of inputs/outputs for replay
- Input Snapshot: Captured inputs for reproducibility
- Processed IDs: Track completed items to skip on re-run

LangGraph Comparison:
- Both require custom implementation for idempotency
- LangGraph checkpointer provides some replay capability
- CrewAI @persist provides state recovery, not full replay
""")

    # Run all demos
    demo_idempotency()
    demo_replay_logging()
    demo_deterministic_workflow()
    demo_idempotent_tool()
    demo_llm_determinism_config()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
