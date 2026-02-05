"""
07_durable_basic.py - Durable Execution Basics (DU-01 to DU-06)

Purpose: Verify state management in CrewAI Flow
- Basic usage of Flow
- Step chaining with @start, @listen
- State management
- DU-05: TTL and automatic state cleanup
- DU-06: Concurrent access control

LangGraph Comparison:
- LangGraph: Explicitly configure Checkpointer (Memory/SQLite/Postgres)
- CrewAI: State management with Flow + Pydantic state
"""

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field


# DB configuration for persistence
DB_PATH = Path("./db")
DB_PATH.mkdir(exist_ok=True)


# =============================================================================
# State TTL Management (DU-05)
# =============================================================================

class StateTTLManager:
    """
    Manages state with Time-To-Live (TTL) and automatic cleanup (DU-05).

    States older than TTL are automatically cleaned up.
    """

    def __init__(self, storage_path: Path, default_ttl_hours: int = 24):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.default_ttl_hours = default_ttl_hours
        self.metadata_file = storage_path / "state_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load state metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save state metadata to file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def register_state(
        self,
        state_id: str,
        ttl_hours: Optional[int] = None,
    ):
        """Register a state with TTL."""
        ttl = ttl_hours or self.default_ttl_hours
        expires_at = datetime.now() + timedelta(hours=ttl)

        self._metadata[state_id] = {
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat(),
            "ttl_hours": ttl,
            "accessed_at": datetime.now().isoformat(),
        }
        self._save_metadata()
        print(f"[TTL] Registered state {state_id}, expires in {ttl}h")

    def touch(self, state_id: str):
        """Update last access time for a state."""
        if state_id in self._metadata:
            self._metadata[state_id]["accessed_at"] = datetime.now().isoformat()
            self._save_metadata()

    def is_expired(self, state_id: str) -> bool:
        """Check if a state has expired."""
        if state_id not in self._metadata:
            return True

        expires_at = datetime.fromisoformat(self._metadata[state_id]["expires_at"])
        return datetime.now() > expires_at

    def cleanup_expired(self) -> list[str]:
        """Remove expired states and return their IDs."""
        expired = []
        for state_id, meta in list(self._metadata.items()):
            expires_at = datetime.fromisoformat(meta["expires_at"])
            if datetime.now() > expires_at:
                expired.append(state_id)
                del self._metadata[state_id]

                # Delete state file if exists
                state_file = self.storage_path / f"{state_id}.json"
                if state_file.exists():
                    state_file.unlink()
                    print(f"[TTL] Cleaned up expired state: {state_id}")

        if expired:
            self._save_metadata()

        return expired

    def extend_ttl(self, state_id: str, additional_hours: int):
        """Extend TTL for a state."""
        if state_id not in self._metadata:
            return

        current_expires = datetime.fromisoformat(self._metadata[state_id]["expires_at"])
        new_expires = current_expires + timedelta(hours=additional_hours)
        self._metadata[state_id]["expires_at"] = new_expires.isoformat()
        self._save_metadata()
        print(f"[TTL] Extended {state_id} TTL by {additional_hours}h")

    def get_state_info(self, state_id: str) -> Optional[dict]:
        """Get state metadata."""
        return self._metadata.get(state_id)


# =============================================================================
# Concurrent Access Control (DU-06)
# =============================================================================

class ConcurrencyManager:
    """
    Manages concurrent access to workflow states (DU-06).

    Prevents race conditions when multiple processes access the same state.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._lock_file_path = storage_path / "locks.json"
        self._active_locks: dict[str, dict] = {}

    def _get_lock(self, state_id: str) -> threading.Lock:
        """Get or create a lock for a state."""
        with self._global_lock:
            if state_id not in self._locks:
                self._locks[state_id] = threading.Lock()
            return self._locks[state_id]

    def acquire(self, state_id: str, timeout: float = 30.0) -> bool:
        """
        Acquire a lock for a state.

        Returns True if lock acquired, False if timeout.
        """
        lock = self._get_lock(state_id)
        acquired = lock.acquire(timeout=timeout)

        if acquired:
            self._active_locks[state_id] = {
                "acquired_at": datetime.now().isoformat(),
                "thread_id": threading.current_thread().ident,
            }
            print(f"[Lock] Acquired lock for {state_id}")

        return acquired

    def release(self, state_id: str):
        """Release a lock for a state."""
        if state_id in self._locks:
            try:
                self._locks[state_id].release()
                if state_id in self._active_locks:
                    del self._active_locks[state_id]
                print(f"[Lock] Released lock for {state_id}")
            except RuntimeError:
                pass  # Lock not held

    def is_locked(self, state_id: str) -> bool:
        """Check if a state is currently locked."""
        if state_id not in self._locks:
            return False
        return self._locks[state_id].locked()

    def get_active_locks(self) -> dict:
        """Get all active locks."""
        return self._active_locks.copy()

    class LockContext:
        """Context manager for state locking."""

        def __init__(self, manager: 'ConcurrencyManager', state_id: str, timeout: float = 30.0):
            self.manager = manager
            self.state_id = state_id
            self.timeout = timeout
            self.acquired = False

        def __enter__(self):
            self.acquired = self.manager.acquire(self.state_id, self.timeout)
            if not self.acquired:
                raise TimeoutError(f"Could not acquire lock for {self.state_id}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.acquired:
                self.manager.release(self.state_id)
            return False

    def lock(self, state_id: str, timeout: float = 30.0) -> LockContext:
        """Get a lock context manager."""
        return self.LockContext(self, state_id, timeout)


# =============================================================================
# Enhanced Workflow State
# =============================================================================

class WorkflowState(BaseModel):
    """State for the workflow."""

    workflow_id: str = ""
    current_step: str = "init"
    data: dict = {}
    history: list = []
    created_at: str = ""
    updated_at: str = ""
    # DU-05: TTL fields
    ttl_hours: int = Field(default=24, description="Time-to-live in hours")
    expires_at: str = ""


class SimpleWorkflow(Flow[WorkflowState]):
    """
    A simple workflow demonstrating CrewAI Flow.

    Flow steps are connected via @start and @listen decorators.
    @listen uses method name (not return value) to chain steps.
    """

    @start()
    def step1_initialize(self):
        """Initialize the workflow."""
        print("\n[Step 1] Initializing workflow...")

        self.state.workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state.current_step = "initialized"
        self.state.created_at = datetime.now().isoformat()
        self.state.updated_at = datetime.now().isoformat()
        self.state.history.append({
            "step": "initialize",
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })

        print(f"[Step 1] Workflow ID: {self.state.workflow_id}")

    @listen(step1_initialize)
    def step2_gather_data(self):
        """Gather data step."""
        print("\n[Step 2] Gathering data...")

        # Simulate data gathering
        self.state.data = {
            "source": "api",
            "records": 100,
            "quality_score": 0.95
        }
        self.state.current_step = "data_gathered"
        self.state.updated_at = datetime.now().isoformat()
        self.state.history.append({
            "step": "gather_data",
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "records": 100
        })

        print(f"[Step 2] Gathered {self.state.data['records']} records")

    @listen(step2_gather_data)
    def step3_process_data(self):
        """Process the gathered data."""
        print("\n[Step 3] Processing data...")

        # Simulate processing
        self.state.data["processed"] = True
        self.state.data["analysis"] = {
            "mean": 42.5,
            "median": 40.0,
            "std_dev": 5.2
        }
        self.state.current_step = "data_processed"
        self.state.updated_at = datetime.now().isoformat()
        self.state.history.append({
            "step": "process_data",
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })

        print(f"[Step 3] Processing complete. Analysis: {self.state.data['analysis']}")

    @listen(step3_process_data)
    def step4_generate_report(self):
        """Generate final report."""
        print("\n[Step 4] Generating report...")

        self.state.current_step = "completed"
        self.state.updated_at = datetime.now().isoformat()
        self.state.history.append({
            "step": "generate_report",
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })

        report = f"""
Workflow Report
===============
ID: {self.state.workflow_id}
Created: {self.state.created_at}
Completed: {self.state.updated_at}

Data Summary:
- Records: {self.state.data.get('records', 'N/A')}
- Quality Score: {self.state.data.get('quality_score', 'N/A')}

Analysis Results:
- Mean: {self.state.data.get('analysis', {}).get('mean', 'N/A')}
- Median: {self.state.data.get('analysis', {}).get('median', 'N/A')}
- Std Dev: {self.state.data.get('analysis', {}).get('std_dev', 'N/A')}

Execution History:
{chr(10).join([f"  - {h['step']}: {h['status']} at {h['timestamp']}" for h in self.state.history])}
"""
        print(report)
        return report


# =============================================================================
# Demonstrations
# =============================================================================

def demo_basic_flow():
    """Demonstrate basic Flow execution."""
    print("=" * 60)
    print("Demo 1: Basic Flow Execution")
    print("=" * 60)

    flow = SimpleWorkflow()
    result = flow.kickoff()

    print("\n" + "=" * 60)
    print("Workflow Result:")
    print("=" * 60)
    print(result)

    # Display final state
    print("\n" + "=" * 60)
    print("Final State Object:")
    print("=" * 60)
    print(f"Workflow ID: {flow.state.workflow_id}")
    print(f"Current Step: {flow.state.current_step}")
    print(f"History Length: {len(flow.state.history)}")


def demo_ttl_management():
    """Demonstrate TTL and automatic cleanup (DU-05)."""
    print("\n" + "=" * 60)
    print("Demo 2: State TTL Management (DU-05)")
    print("=" * 60)

    ttl_manager = StateTTLManager(DB_PATH / "ttl_states", default_ttl_hours=1)

    # Register states with different TTLs
    ttl_manager.register_state("state_1", ttl_hours=1)
    ttl_manager.register_state("state_2", ttl_hours=24)
    ttl_manager.register_state("state_3", ttl_hours=0)  # Immediately expired

    # Check states
    print("\nState status:")
    for state_id in ["state_1", "state_2", "state_3"]:
        info = ttl_manager.get_state_info(state_id)
        expired = ttl_manager.is_expired(state_id)
        if info:
            print(f"  {state_id}: expires={info['expires_at']}, expired={expired}")

    # Cleanup expired
    print("\nCleaning up expired states...")
    expired = ttl_manager.cleanup_expired()
    print(f"Cleaned up: {expired}")

    # Extend TTL
    print("\nExtending TTL for state_1...")
    ttl_manager.extend_ttl("state_1", 12)
    info = ttl_manager.get_state_info("state_1")
    print(f"New expiry: {info['expires_at']}")


def demo_concurrent_access():
    """Demonstrate concurrent access control (DU-06)."""
    print("\n" + "=" * 60)
    print("Demo 3: Concurrent Access Control (DU-06)")
    print("=" * 60)

    manager = ConcurrencyManager(DB_PATH / "locks")
    state_id = "shared_state"

    # Basic lock/unlock
    print("\n--- Basic Lock/Unlock ---")
    print(f"Is locked: {manager.is_locked(state_id)}")
    manager.acquire(state_id)
    print(f"Is locked: {manager.is_locked(state_id)}")
    manager.release(state_id)
    print(f"Is locked: {manager.is_locked(state_id)}")

    # Context manager
    print("\n--- Context Manager ---")
    with manager.lock(state_id):
        print(f"Inside lock context, is_locked: {manager.is_locked(state_id)}")
    print(f"Outside lock context, is_locked: {manager.is_locked(state_id)}")

    # Concurrent access simulation
    print("\n--- Concurrent Access Simulation ---")
    results = []

    def worker(worker_id: int):
        with manager.lock(state_id, timeout=5.0):
            print(f"  Worker {worker_id}: acquired lock")
            time.sleep(0.1)  # Simulate work
            results.append(worker_id)
            print(f"  Worker {worker_id}: releasing lock")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Execution order: {results}")


def demo_state_persistence_with_ttl():
    """Demonstrate state persistence with TTL."""
    print("\n" + "=" * 60)
    print("Demo 4: State Persistence with TTL")
    print("=" * 60)

    ttl_manager = StateTTLManager(DB_PATH / "persistent_states")
    concurrency_manager = ConcurrencyManager(DB_PATH / "persistent_locks")

    state_id = "persistent_workflow_001"

    # Register and lock
    ttl_manager.register_state(state_id, ttl_hours=48)

    with concurrency_manager.lock(state_id):
        # Simulate state operations
        state_file = DB_PATH / "persistent_states" / f"{state_id}.json"
        state = {
            "id": state_id,
            "status": "in_progress",
            "data": {"items": [1, 2, 3]},
        }
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
        print(f"Saved state to {state_file}")

        # Touch to update access time
        ttl_manager.touch(state_id)

    # Show state info
    info = ttl_manager.get_state_info(state_id)
    print(f"State info: {info}")


def main():
    print("=" * 60)
    print("Durable Execution: Basic Flow Test (DU-01 to DU-06)")
    print("=" * 60)
    print("""
This example demonstrates CrewAI Flow for workflow management.

Key Points:
- @start() marks the entry point
- @listen(method) chains to previous method
- State is managed via Pydantic model

Enhanced Features (DU-05, DU-06):
- DU-05: TTL management for automatic state cleanup
- DU-06: Concurrent access control with locking

LangGraph Comparison:
- CrewAI Flow: Decorator-based, implicit connections
- LangGraph: Explicit add_edge(), more control
- Both need custom TTL and concurrency handling
""")

    # Run all demos
    demo_basic_flow()
    demo_ttl_management()
    demo_concurrent_access()
    demo_state_persistence_with_ttl()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
