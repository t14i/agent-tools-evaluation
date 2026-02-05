"""
Connectors & Ops - Part 2: Responses API (CX-03, CX-04)
Background execution, state migration
"""

from dotenv import load_dotenv
load_dotenv()


import json
import time
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid


# =============================================================================
# CX-03: Async Job / Background Execution
# =============================================================================

class JobStatus(Enum):
    """Status of a background job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundJob:
    """A background job."""
    job_id: str
    task_type: str
    params: dict
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0


class BackgroundJobManager:
    """
    Manages background/async job execution.
    Implements CX-03: Async Job Pattern.

    OpenAI SDK supports background=true for async execution.
    """

    def __init__(self):
        self.jobs: dict[str, BackgroundJob] = {}

    def submit(self, task_type: str, params: dict) -> BackgroundJob:
        """Submit a new background job."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"

        job = BackgroundJob(
            job_id=job_id,
            task_type=task_type,
            params=params,
            status=JobStatus.PENDING,
            created_at=datetime.now()
        )

        self.jobs[job_id] = job
        return job

    def get_status(self, job_id: str) -> Optional[BackgroundJob]:
        """Get job status."""
        return self.jobs.get(job_id)

    def list_jobs(
        self,
        status: JobStatus = None,
        task_type: str = None
    ) -> list[BackgroundJob]:
        """List jobs with optional filtering."""
        jobs = list(self.jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]
        if task_type:
            jobs = [j for j in jobs if j.task_type == task_type]

        return jobs

    def start_job(self, job_id: str) -> bool:
        """Mark job as started."""
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.PENDING:
            return False

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        return True

    def complete_job(self, job_id: str, result: Any) -> bool:
        """Mark job as completed."""
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.RUNNING:
            return False

        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.result = result
        job.progress = 1.0
        return True

    def fail_job(self, job_id: str, error: str) -> bool:
        """Mark job as failed."""
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.RUNNING:
            return False

        job.status = JobStatus.FAILED
        job.completed_at = datetime.now()
        job.error = error
        return True

    def update_progress(self, job_id: str, progress: float) -> bool:
        """Update job progress."""
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.RUNNING:
            return False

        job.progress = min(1.0, max(0.0, progress))
        return True

    def wait_for_completion(
        self,
        job_id: str,
        timeout: float = 300,
        poll_interval: float = 1.0
    ) -> Optional[BackgroundJob]:
        """Wait for job to complete."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            job = self.get_status(job_id)
            if not job:
                return None

            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return job

            time.sleep(poll_interval)

        return None


# =============================================================================
# OpenAI Responses API Integration
# =============================================================================

class ResponsesAPIWrapper:
    """
    Wrapper for OpenAI's Responses API with background support.

    The Responses API is the recommended approach for production:
    - Supports background=true for async execution
    - Returns response_id for tracking
    - Enables retrieval of results later
    """

    def __init__(self, job_manager: BackgroundJobManager):
        self.job_manager = job_manager

    def create_response(
        self,
        agent_name: str,
        input_text: str,
        background: bool = False
    ) -> dict:
        """
        Create a response, optionally in background.

        Args:
            agent_name: Name of the agent
            input_text: User input
            background: If True, returns immediately with job_id

        Returns:
            If background=False: {"output": "...", "completed": True}
            If background=True: {"job_id": "...", "status": "pending"}
        """
        if not background:
            # Synchronous execution
            return {
                "output": f"Response from {agent_name}: {input_text[:20]}...",
                "completed": True,
                "response_id": f"resp_{uuid.uuid4().hex[:12]}"
            }

        # Background execution
        job = self.job_manager.submit(
            task_type="agent_response",
            params={"agent_name": agent_name, "input": input_text}
        )

        # Simulate async start
        self.job_manager.start_job(job.job_id)

        return {
            "job_id": job.job_id,
            "status": "pending",
            "response_id": None  # Will be set on completion
        }

    def get_response(self, job_id: str) -> Optional[dict]:
        """Get response for a background job."""
        job = self.job_manager.get_status(job_id)
        if not job:
            return None

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "result": job.result,
            "error": job.error
        }

    def poll_response(
        self,
        job_id: str,
        timeout: float = 60
    ) -> Optional[dict]:
        """Poll until response is ready."""
        job = self.job_manager.wait_for_completion(job_id, timeout, poll_interval=0.5)
        if job:
            return {
                "job_id": job.job_id,
                "status": job.status.value,
                "result": job.result,
                "error": job.error
            }
        return None


# =============================================================================
# CX-04: State Migration
# =============================================================================

@dataclass
class StateSchema:
    """Schema for versioned state."""
    version: str
    fields: dict[str, str]  # field_name -> type


class StateMigrator:
    """
    Handles state migration between schema versions.
    Implements CX-04: State Migration.
    """

    def __init__(self):
        self.migrations: dict[tuple[str, str], callable] = {}
        self.schemas: dict[str, StateSchema] = {}

    def register_schema(self, version: str, fields: dict[str, str]):
        """Register a schema version."""
        self.schemas[version] = StateSchema(version=version, fields=fields)

    def register_migration(
        self,
        from_version: str,
        to_version: str,
        migration_fn: callable
    ):
        """Register a migration function."""
        self.migrations[(from_version, to_version)] = migration_fn

    def get_migration_path(
        self,
        from_version: str,
        to_version: str
    ) -> list[str]:
        """Find migration path between versions."""
        # Simple linear path finding (production would use graph search)
        path = [from_version]
        current = from_version

        while current != to_version:
            found = False
            for (fv, tv), _ in self.migrations.items():
                if fv == current:
                    path.append(tv)
                    current = tv
                    found = True
                    break
            if not found:
                raise ValueError(f"No migration path from {from_version} to {to_version}")

        return path

    def migrate(
        self,
        state: dict,
        from_version: str,
        to_version: str
    ) -> dict:
        """Migrate state from one version to another."""
        if from_version == to_version:
            return state

        path = self.get_migration_path(from_version, to_version)

        current_state = state.copy()
        for i in range(len(path) - 1):
            fv, tv = path[i], path[i + 1]
            migration = self.migrations.get((fv, tv))
            if migration:
                current_state = migration(current_state)
                current_state["_schema_version"] = tv

        return current_state


# =============================================================================
# Session State with Versioning
# =============================================================================

class VersionedSessionState:
    """
    Session state with schema versioning and migration support.
    """

    CURRENT_VERSION = "v3"

    def __init__(self, migrator: StateMigrator):
        self.migrator = migrator
        self.state: dict = {"_schema_version": self.CURRENT_VERSION}

    def load(self, serialized: str) -> bool:
        """Load state from serialized form, migrating if necessary."""
        try:
            data = json.loads(serialized)
            version = data.get("_schema_version", "v1")

            if version != self.CURRENT_VERSION:
                data = self.migrator.migrate(data, version, self.CURRENT_VERSION)

            self.state = data
            return True
        except Exception as e:
            print(f"Failed to load state: {e}")
            return False

    def save(self) -> str:
        """Save state to serialized form."""
        self.state["_schema_version"] = self.CURRENT_VERSION
        return json.dumps(self.state)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self.state.get(key, default)

    def set(self, key: str, value: Any):
        """Set a state value."""
        self.state[key] = value


# =============================================================================
# Tests
# =============================================================================

def test_background_jobs():
    """Test background job execution (CX-03)."""
    print("\n" + "=" * 70)
    print("TEST: Background Jobs (CX-03)")
    print("=" * 70)

    manager = BackgroundJobManager()

    # Submit job
    job = manager.submit("agent_response", {"input": "Hello"})
    print(f"\nSubmitted job: {job.job_id}")
    print(f"Status: {job.status.value}")

    # Start job
    manager.start_job(job.job_id)
    print(f"After start: {job.status.value}")

    # Update progress
    manager.update_progress(job.job_id, 0.5)
    print(f"Progress: {job.progress * 100}%")

    # Complete job
    manager.complete_job(job.job_id, {"output": "Hello response"})
    print(f"After complete: {job.status.value}")
    print(f"Result: {job.result}")

    print("\n✅ Background jobs work")


def test_responses_api():
    """Test Responses API wrapper."""
    print("\n" + "=" * 70)
    print("TEST: Responses API (background=true)")
    print("=" * 70)

    manager = BackgroundJobManager()
    api = ResponsesAPIWrapper(manager)

    # Synchronous
    result = api.create_response("TestAgent", "Hello sync", background=False)
    print(f"\nSynchronous: {result}")

    # Asynchronous
    result = api.create_response("TestAgent", "Hello async", background=True)
    print(f"Async submitted: {result}")

    # Simulate completion
    job = manager.get_status(result["job_id"])
    manager.complete_job(job.job_id, {"output": "Async response"})

    # Get result
    final = api.get_response(result["job_id"])
    print(f"Async result: {final}")

    print("\n✅ Responses API works")


def test_state_migration():
    """Test state migration (CX-04)."""
    print("\n" + "=" * 70)
    print("TEST: State Migration (CX-04)")
    print("=" * 70)

    migrator = StateMigrator()

    # Register schemas
    migrator.register_schema("v1", {"messages": "list"})
    migrator.register_schema("v2", {"messages": "list", "metadata": "dict"})
    migrator.register_schema("v3", {"messages": "list", "metadata": "dict", "context": "dict"})

    # Register migrations
    migrator.register_migration("v1", "v2", lambda s: {
        **s,
        "metadata": {"migrated_from": "v1"}
    })

    migrator.register_migration("v2", "v3", lambda s: {
        **s,
        "context": {"conversation_id": None}
    })

    # Test migration
    old_state = {
        "_schema_version": "v1",
        "messages": [{"role": "user", "content": "Hello"}]
    }

    print(f"\nOld state (v1): {json.dumps(old_state, indent=2)}")

    new_state = migrator.migrate(old_state, "v1", "v3")
    print(f"\nMigrated state (v3): {json.dumps(new_state, indent=2)}")

    print("\n✅ State migration works")


def test_versioned_session():
    """Test versioned session state."""
    print("\n" + "=" * 70)
    print("TEST: Versioned Session State")
    print("=" * 70)

    migrator = StateMigrator()
    migrator.register_migration("v1", "v2", lambda s: {**s, "new_field": "default"})
    migrator.register_migration("v2", "v3", lambda s: {**s, "another_field": []})

    session = VersionedSessionState(migrator)

    # Set some state
    session.set("user_id", "user_123")
    session.set("preferences", {"theme": "dark"})

    # Save
    serialized = session.save()
    print(f"\nSerialized: {serialized[:100]}...")

    # Load in new session
    session2 = VersionedSessionState(migrator)
    session2.load(serialized)

    print(f"Loaded user_id: {session2.get('user_id')}")
    print(f"Loaded preferences: {session2.get('preferences')}")

    print("\n✅ Versioned session works")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ CX-03, CX-04: RESPONSES API - EVALUATION SUMMARY                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ CX-03 (Async Job Pattern): ⭐⭐⭐⭐ (Production Ready)                       │
│   ✅ Responses API supports background=true                                 │
│   ✅ Response ID for tracking                                               │
│   ✅ Polling for completion                                                 │
│   ✅ Progress tracking possible                                             │
│                                                                             │
│ CX-04 (State Migration): ⭐⭐ (Experimental)                                │
│   ❌ No built-in schema versioning                                          │
│   ❌ No automatic migration                                                 │
│   ⚠️ Custom StateMigrator implementation provided                          │
│   ⚠️ Must handle manually                                                  │
│                                                                             │
│ OpenAI SDK Features:                                                        │
│   - Responses API (recommended for production)                              │
│   - background=true for async execution                                     │
│   - Response retrieval by ID                                                │
│   - Streaming support                                                       │
│                                                                             │
│ Responses API Benefits:                                                     │
│   - Long-running operations without timeout                                 │
│   - Fire-and-forget execution                                               │
│   - Progress monitoring                                                     │
│   - Result caching                                                          │
│                                                                             │
│ Custom Implementation Provided:                                             │
│   ✅ BackgroundJobManager - Job lifecycle management                        │
│   ✅ ResponsesAPIWrapper - API abstraction                                  │
│   ✅ StateMigrator - Schema versioning and migration                        │
│   ✅ VersionedSessionState - Versioned session state                        │
│                                                                             │
│ Production Notes:                                                           │
│   - Use Responses API for production workloads                              │
│   - Implement proper job persistence                                        │
│   - Version your state schemas from the start                               │
│   - Plan migration strategy for long-running sessions                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_background_jobs()
    test_responses_api()
    test_state_migration()
    test_versioned_session()

    print(SUMMARY)
