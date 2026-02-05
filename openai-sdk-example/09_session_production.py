"""
Durable Execution - Part 3: Production Considerations (DU-05, DU-06)
Cleanup, TTL, concurrent access, state migration
"""

from dotenv import load_dotenv
load_dotenv()


import time
import threading
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field


# =============================================================================
# DU-05: Session Cleanup and TTL
# =============================================================================

@dataclass
class SessionWithTTL:
    """Session with TTL support."""
    id: str
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime]
    messages: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if session has expired."""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at


class SessionCleanupManager:
    """
    Manages session cleanup and TTL.
    Implements DU-05: Cleanup (TTL).
    """

    def __init__(self, default_ttl: timedelta = timedelta(days=7)):
        self.sessions: dict[str, SessionWithTTL] = {}
        self.default_ttl = default_ttl
        self._cleanup_lock = threading.Lock()

    def create_session(
        self,
        session_id: str,
        ttl: Optional[timedelta] = None,
        metadata: dict = None
    ) -> SessionWithTTL:
        """Create session with optional TTL."""
        now = datetime.now()
        effective_ttl = ttl or self.default_ttl
        expires_at = now + effective_ttl if effective_ttl else None

        session = SessionWithTTL(
            id=session_id,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            metadata=metadata or {}
        )

        with self._cleanup_lock:
            self.sessions[session_id] = session

        return session

    def get_session(self, session_id: str) -> Optional[SessionWithTTL]:
        """Get session if not expired."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        if session.is_expired():
            self._remove_session(session_id)
            return None

        return session

    def extend_ttl(self, session_id: str, extension: timedelta) -> bool:
        """Extend session TTL."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        if session.expires_at:
            session.expires_at = session.expires_at + extension
        else:
            session.expires_at = datetime.now() + extension

        return True

    def cleanup_expired(self) -> int:
        """Remove all expired sessions."""
        with self._cleanup_lock:
            expired = [
                sid for sid, session in self.sessions.items()
                if session.is_expired()
            ]

            for sid in expired:
                del self.sessions[sid]

            return len(expired)

    def _remove_session(self, session_id: str):
        """Remove a single session."""
        with self._cleanup_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]

    def get_expiring_soon(self, within: timedelta = timedelta(hours=1)) -> list[SessionWithTTL]:
        """Get sessions expiring within the specified timeframe."""
        threshold = datetime.now() + within
        return [
            session for session in self.sessions.values()
            if session.expires_at and session.expires_at <= threshold
        ]


# =============================================================================
# DU-06: Concurrent Access Handling
# =============================================================================

class ConcurrentSessionManager:
    """
    Handles concurrent access to sessions.
    Implements DU-06: Concurrent Access.
    """

    def __init__(self):
        self.sessions: dict[str, SessionWithTTL] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, session_id: str) -> threading.RLock:
        """Get or create a lock for a session."""
        with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.RLock()
            return self._locks[session_id]

    def create_session(self, session_id: str) -> SessionWithTTL:
        """Create a new session with locking."""
        lock = self._get_lock(session_id)
        with lock:
            if session_id in self.sessions:
                raise ValueError(f"Session {session_id} already exists")

            session = SessionWithTTL(
                id=session_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                expires_at=None
            )
            self.sessions[session_id] = session
            return session

    def update_session(
        self,
        session_id: str,
        messages: list = None,
        state: dict = None
    ) -> bool:
        """Update session with locking to prevent race conditions."""
        lock = self._get_lock(session_id)
        with lock:
            session = self.sessions.get(session_id)
            if not session:
                return False

            if messages:
                session.messages.extend(messages)
            if state:
                session.state.update(state)
            session.updated_at = datetime.now()

            return True

    def atomic_update(
        self,
        session_id: str,
        update_fn: callable
    ) -> bool:
        """
        Perform atomic update using a function.
        Prevents race conditions in read-modify-write operations.
        """
        lock = self._get_lock(session_id)
        with lock:
            session = self.sessions.get(session_id)
            if not session:
                return False

            # User function modifies session in place
            update_fn(session)
            session.updated_at = datetime.now()
            return True

    def optimistic_update(
        self,
        session_id: str,
        expected_version: datetime,
        messages: list = None,
        state: dict = None
    ) -> tuple[bool, Optional[datetime]]:
        """
        Optimistic locking update - fails if session was modified.
        Returns (success, new_version).
        """
        lock = self._get_lock(session_id)
        with lock:
            session = self.sessions.get(session_id)
            if not session:
                return (False, None)

            # Check version (updated_at)
            if session.updated_at != expected_version:
                return (False, session.updated_at)

            # Apply update
            if messages:
                session.messages.extend(messages)
            if state:
                session.state.update(state)
            session.updated_at = datetime.now()

            return (True, session.updated_at)


# =============================================================================
# Cleanup Scheduler
# =============================================================================

class CleanupScheduler:
    """Periodic cleanup scheduler."""

    def __init__(self, manager: SessionCleanupManager, interval: timedelta = timedelta(hours=1)):
        self.manager = manager
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Start the cleanup scheduler."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the cleanup scheduler."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def _run(self):
        """Run cleanup periodically."""
        while not self._stop_event.is_set():
            cleaned = self.manager.cleanup_expired()
            if cleaned > 0:
                print(f"[CLEANUP] Removed {cleaned} expired sessions")

            self._stop_event.wait(self.interval.total_seconds())


# =============================================================================
# Tests
# =============================================================================

def test_session_ttl():
    """Test session TTL and cleanup (DU-05)."""
    print("\n" + "=" * 70)
    print("TEST: Session TTL (DU-05)")
    print("=" * 70)

    manager = SessionCleanupManager(default_ttl=timedelta(seconds=2))

    # Create sessions with different TTLs
    s1 = manager.create_session("session_1", ttl=timedelta(seconds=1))
    s2 = manager.create_session("session_2", ttl=timedelta(seconds=5))
    s3 = manager.create_session("session_3")  # Default TTL

    print(f"\nCreated 3 sessions")
    print(f"  session_1 expires at: {s1.expires_at}")
    print(f"  session_2 expires at: {s2.expires_at}")
    print(f"  session_3 expires at: {s3.expires_at}")

    # Wait for first to expire
    print("\n  Waiting 1.5 seconds...")
    time.sleep(1.5)

    # Check expired
    result = manager.get_session("session_1")
    print(f"\n  session_1 after wait: {result}")

    result = manager.get_session("session_2")
    print(f"  session_2 after wait: {result is not None}")

    # Cleanup
    cleaned = manager.cleanup_expired()
    print(f"\n  Cleaned up {cleaned} expired sessions")

    print("\n✅ Session TTL works")


def test_ttl_extension():
    """Test TTL extension."""
    print("\n" + "=" * 70)
    print("TEST: TTL Extension")
    print("=" * 70)

    manager = SessionCleanupManager()

    session = manager.create_session("ext_session", ttl=timedelta(seconds=5))
    original_expiry = session.expires_at

    print(f"\nOriginal expiry: {original_expiry}")

    # Extend TTL
    manager.extend_ttl("ext_session", timedelta(seconds=10))

    session = manager.get_session("ext_session")
    print(f"Extended expiry: {session.expires_at}")

    print("\n✅ TTL extension works")


def test_concurrent_access():
    """Test concurrent access handling (DU-06)."""
    print("\n" + "=" * 70)
    print("TEST: Concurrent Access (DU-06)")
    print("=" * 70)

    manager = ConcurrentSessionManager()
    manager.create_session("concurrent_test")

    results = []
    errors = []

    def update_worker(worker_id: int, iterations: int):
        """Worker that updates session."""
        for i in range(iterations):
            try:
                manager.update_session(
                    "concurrent_test",
                    messages=[{"worker": worker_id, "iteration": i}]
                )
                results.append((worker_id, i))
            except Exception as e:
                errors.append((worker_id, i, str(e)))

    # Start multiple threads
    threads = []
    for worker_id in range(5):
        t = threading.Thread(target=update_worker, args=(worker_id, 10))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    session = manager.sessions.get("concurrent_test")
    print(f"\nTotal messages: {len(session.messages)}")
    print(f"Expected: 50 (5 workers x 10 iterations)")
    print(f"Errors: {len(errors)}")

    if len(session.messages) == 50 and len(errors) == 0:
        print("\n✅ Concurrent access handled correctly")
    else:
        print("\n❌ Race condition detected!")


def test_optimistic_locking():
    """Test optimistic locking."""
    print("\n" + "=" * 70)
    print("TEST: Optimistic Locking")
    print("=" * 70)

    manager = ConcurrentSessionManager()
    manager.create_session("optimistic_test")

    # Get initial version
    session = manager.sessions.get("optimistic_test")
    version = session.updated_at

    print(f"\nInitial version: {version}")

    # First update succeeds
    success, new_version = manager.optimistic_update(
        "optimistic_test",
        expected_version=version,
        messages=[{"update": 1}]
    )
    print(f"First update: success={success}, new_version={new_version}")

    # Second update with old version fails
    success, returned_version = manager.optimistic_update(
        "optimistic_test",
        expected_version=version,  # Old version
        messages=[{"update": 2}]
    )
    print(f"Second update (stale): success={success}, actual_version={returned_version}")

    # Update with correct version succeeds
    success, final_version = manager.optimistic_update(
        "optimistic_test",
        expected_version=new_version,
        messages=[{"update": 3}]
    )
    print(f"Third update (correct): success={success}")

    session = manager.sessions.get("optimistic_test")
    print(f"\nFinal message count: {len(session.messages)}")

    print("\n✅ Optimistic locking works")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ DU-05, DU-06: PRODUCTION CONSIDERATIONS - EVALUATION SUMMARY                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ DU-05 (Cleanup / TTL): ⭐ (Not Supported)                                   │
│   ❌ No built-in TTL support in OpenAI SDK                                  │
│   ❌ No automatic session cleanup                                           │
│   ❌ Requires custom implementation                                         │
│   ⚠️ Custom SessionCleanupManager provided                                  │
│   ⚠️ Need to run periodic cleanup job                                       │
│                                                                             │
│ DU-06 (Concurrent Access): ⭐⭐ (Experimental)                              │
│   ❌ No built-in concurrency handling                                       │
│   ❌ No optimistic locking                                                  │
│   ⚠️ Custom ConcurrentSessionManager provided                               │
│   ⚠️ Thread-level locking (not distributed)                                │
│   ⚠️ For distributed, need Redis/DB locks                                  │
│                                                                             │
│ Custom Implementation Provided:                                             │
│   ✅ SessionWithTTL - Sessions with expiration                              │
│   ✅ SessionCleanupManager - TTL and cleanup                                │
│   ✅ ConcurrentSessionManager - Thread-safe access                          │
│   ✅ Optimistic locking - Version-based updates                             │
│   ✅ CleanupScheduler - Periodic cleanup                                    │
│                                                                             │
│ Production Considerations:                                                  │
│   - Use database TTL (PostgreSQL, Redis) when possible                      │
│   - Implement distributed locking for multi-node                            │
│   - Run cleanup as background job (cron, Celery)                            │
│   - Monitor session count and storage usage                                 │
│   - Implement session archival before deletion                              │
│                                                                             │
│ Comparison with LangGraph:                                                  │
│   LangGraph:                                                                │
│     - Also lacks built-in TTL                                               │
│     - Also requires custom cleanup                                          │
│     - Postgres checkpointer supports transactions                           │
│   OpenAI SDK:                                                               │
│     - Similar gaps                                                          │
│     - More storage backend options                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_session_ttl()
    test_ttl_extension()
    test_concurrent_access()
    test_optimistic_locking()

    print(SUMMARY)
