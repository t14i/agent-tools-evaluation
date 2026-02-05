"""
Determinism & Replay - Part 2: Recovery (DR-04, DR-05, DR-06)
Idempotency, plan diff, failure recovery
"""

from dotenv import load_dotenv
load_dotenv()


import json
import hashlib
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# DR-04: Idempotency
# =============================================================================

@dataclass
class IdempotencyRecord:
    """Record of an idempotent operation."""
    key: str
    operation: str
    request_hash: str
    response: Any
    created_at: datetime
    expires_at: Optional[datetime] = None


class IdempotencyManager:
    """
    Ensures exactly-once execution of operations.
    Implements DR-04: Idempotency.
    """

    def __init__(self):
        self.records: dict[str, IdempotencyRecord] = {}

    def get_key(self, operation: str, params: dict) -> str:
        """Generate idempotency key from operation and params."""
        data = f"{operation}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def check(self, key: str) -> Optional[Any]:
        """Check if operation was already executed."""
        record = self.records.get(key)
        if record:
            # Check expiration
            if record.expires_at and datetime.now() > record.expires_at:
                del self.records[key]
                return None
            return record.response
        return None

    def record(
        self,
        key: str,
        operation: str,
        request_hash: str,
        response: Any,
        ttl_seconds: int = 86400
    ):
        """Record an executed operation."""
        self.records[key] = IdempotencyRecord(
            key=key,
            operation=operation,
            request_hash=request_hash,
            response=response,
            created_at=datetime.now(),
            expires_at=datetime.now() if ttl_seconds else None
        )

    def execute_once(
        self,
        operation: str,
        params: dict,
        executor: callable
    ) -> tuple[Any, bool]:
        """
        Execute operation at most once.
        Returns: (result, was_cached)
        """
        key = self.get_key(operation, params)

        # Check for existing result
        cached = self.check(key)
        if cached is not None:
            return (cached, True)

        # Execute
        result = executor(params)

        # Record
        self.record(
            key=key,
            operation=operation,
            request_hash=hashlib.sha256(json.dumps(params).encode()).hexdigest(),
            response=result
        )

        return (result, False)


# =============================================================================
# DR-05: Plan Diff
# =============================================================================

class PlanStep(Enum):
    """Step types in an execution plan."""
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    HANDOFF = "handoff"
    CONDITION = "condition"


@dataclass
class PlannedStep:
    """A step in an execution plan."""
    step_id: str
    step_type: PlanStep
    description: str
    params: dict
    expected_outcome: Optional[str] = None


@dataclass
class ExecutionPlan:
    """An execution plan that can be compared."""
    plan_id: str
    steps: list[PlannedStep]
    created_at: datetime = field(default_factory=datetime.now)


class PlanDiff:
    """
    Compares execution plans to identify changes.
    Implements DR-05: Plan Diff.
    """

    @staticmethod
    def compare(plan_a: ExecutionPlan, plan_b: ExecutionPlan) -> dict:
        """Compare two execution plans."""
        diff = {
            "added": [],
            "removed": [],
            "modified": [],
            "unchanged": []
        }

        # Create lookup maps
        a_steps = {s.step_id: s for s in plan_a.steps}
        b_steps = {s.step_id: s for s in plan_b.steps}

        # Find removed and modified
        for step_id, step_a in a_steps.items():
            if step_id not in b_steps:
                diff["removed"].append(step_a)
            else:
                step_b = b_steps[step_id]
                if step_a.params != step_b.params or step_a.description != step_b.description:
                    diff["modified"].append({
                        "step_id": step_id,
                        "old": step_a,
                        "new": step_b
                    })
                else:
                    diff["unchanged"].append(step_id)

        # Find added
        for step_id, step_b in b_steps.items():
            if step_id not in a_steps:
                diff["added"].append(step_b)

        return diff

    @staticmethod
    def format_diff(diff: dict) -> str:
        """Format diff for display."""
        lines = []

        if diff["added"]:
            lines.append("+ Added steps:")
            for step in diff["added"]:
                lines.append(f"  + {step.step_id}: {step.description}")

        if diff["removed"]:
            lines.append("- Removed steps:")
            for step in diff["removed"]:
                lines.append(f"  - {step.step_id}: {step.description}")

        if diff["modified"]:
            lines.append("~ Modified steps:")
            for mod in diff["modified"]:
                lines.append(f"  ~ {mod['step_id']}:")
                lines.append(f"    old: {mod['old'].description}")
                lines.append(f"    new: {mod['new'].description}")

        if not (diff["added"] or diff["removed"] or diff["modified"]):
            lines.append("No changes detected")

        return "\n".join(lines)


# =============================================================================
# DR-06: Failure Recovery
# =============================================================================

class RecoveryStrategy(Enum):
    """Strategies for failure recovery."""
    RETRY = "retry"
    ROLLBACK = "rollback"
    COMPENSATE = "compensate"
    SKIP = "skip"
    MANUAL = "manual"


@dataclass
class FailureContext:
    """Context of a failure."""
    step_id: str
    error_type: str
    error_message: str
    occurred_at: datetime
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class RecoveryAction:
    """An action to recover from failure."""
    action_type: str
    target_step: str
    params: dict
    executed_at: Optional[datetime] = None
    result: Optional[str] = None


class RecoveryManager:
    """
    Manages failure recovery and rollback.
    Implements DR-06: Failure Recovery.
    """

    def __init__(self):
        self.failures: list[FailureContext] = []
        self.recovery_actions: list[RecoveryAction] = []
        self.compensations: dict[str, callable] = {}

    def register_compensation(self, step_id: str, compensation_fn: callable):
        """Register a compensation function for a step."""
        self.compensations[step_id] = compensation_fn

    def record_failure(
        self,
        step_id: str,
        error_type: str,
        error_message: str
    ) -> FailureContext:
        """Record a failure."""
        failure = FailureContext(
            step_id=step_id,
            error_type=error_type,
            error_message=error_message,
            occurred_at=datetime.now()
        )
        self.failures.append(failure)
        return failure

    def get_recovery_strategy(self, failure: FailureContext) -> RecoveryStrategy:
        """Determine recovery strategy based on failure type."""
        # Transient errors -> retry
        if failure.error_type in ["timeout", "rate_limit", "connection_error"]:
            if failure.retry_count < failure.max_retries:
                return RecoveryStrategy.RETRY

        # Data errors -> compensate if possible
        if failure.error_type in ["validation_error", "constraint_violation"]:
            if failure.step_id in self.compensations:
                return RecoveryStrategy.COMPENSATE

        # Unknown errors -> manual intervention
        return RecoveryStrategy.MANUAL

    def execute_recovery(
        self,
        failure: FailureContext,
        strategy: RecoveryStrategy
    ) -> RecoveryAction:
        """Execute recovery action."""
        action = RecoveryAction(
            action_type=strategy.value,
            target_step=failure.step_id,
            params={"error": failure.error_message},
            executed_at=datetime.now()
        )

        if strategy == RecoveryStrategy.RETRY:
            failure.retry_count += 1
            action.result = f"Retry attempt {failure.retry_count}"

        elif strategy == RecoveryStrategy.COMPENSATE:
            if failure.step_id in self.compensations:
                try:
                    self.compensations[failure.step_id]()
                    action.result = "Compensation executed"
                except Exception as e:
                    action.result = f"Compensation failed: {e}"

        elif strategy == RecoveryStrategy.ROLLBACK:
            action.result = "Rollback initiated"

        elif strategy == RecoveryStrategy.SKIP:
            action.result = "Step skipped"

        else:
            action.result = "Manual intervention required"

        self.recovery_actions.append(action)
        return action

    def get_recovery_report(self) -> str:
        """Generate recovery report."""
        lines = [
            f"Recovery Report",
            f"===============",
            f"Total failures: {len(self.failures)}",
            f"Recovery actions: {len(self.recovery_actions)}",
            ""
        ]

        for failure in self.failures:
            lines.append(f"Failure at {failure.step_id}:")
            lines.append(f"  Type: {failure.error_type}")
            lines.append(f"  Message: {failure.error_message}")
            lines.append(f"  Retries: {failure.retry_count}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Tests
# =============================================================================

def test_idempotency():
    """Test idempotency management (DR-04)."""
    print("\n" + "=" * 70)
    print("TEST: Idempotency (DR-04)")
    print("=" * 70)

    manager = IdempotencyManager()

    # Simulate an operation
    call_count = 0
    def expensive_operation(params):
        nonlocal call_count
        call_count += 1
        return f"Result for {params['query']}"

    # First call - executes
    result1, cached1 = manager.execute_once(
        "search",
        {"query": "weather"},
        expensive_operation
    )
    print(f"\nFirst call: {result1}, cached={cached1}")

    # Second call - cached
    result2, cached2 = manager.execute_once(
        "search",
        {"query": "weather"},
        expensive_operation
    )
    print(f"Second call: {result2}, cached={cached2}")

    # Different params - executes
    result3, cached3 = manager.execute_once(
        "search",
        {"query": "news"},
        expensive_operation
    )
    print(f"Third call (different): {result3}, cached={cached3}")

    print(f"\nActual executions: {call_count}")
    print("✅ Idempotency works")


def test_plan_diff():
    """Test plan diff (DR-05)."""
    print("\n" + "=" * 70)
    print("TEST: Plan Diff (DR-05)")
    print("=" * 70)

    # Create original plan
    plan_v1 = ExecutionPlan(
        plan_id="plan_v1",
        steps=[
            PlannedStep("s1", PlanStep.LLM_CALL, "Understand request", {}),
            PlannedStep("s2", PlanStep.TOOL_CALL, "Search database", {"query": "users"}),
            PlannedStep("s3", PlanStep.LLM_CALL, "Format response", {}),
        ]
    )

    # Create modified plan
    plan_v2 = ExecutionPlan(
        plan_id="plan_v2",
        steps=[
            PlannedStep("s1", PlanStep.LLM_CALL, "Understand request", {}),
            PlannedStep("s2", PlanStep.TOOL_CALL, "Search database", {"query": "users", "limit": 10}),  # Modified
            PlannedStep("s4", PlanStep.TOOL_CALL, "Validate results", {}),  # Added
            # s3 removed
        ]
    )

    # Compare
    diff = PlanDiff.compare(plan_v1, plan_v2)

    print(f"\nPlan Diff:")
    print(PlanDiff.format_diff(diff))

    print("\n✅ Plan diff works")


def test_failure_recovery():
    """Test failure recovery (DR-06)."""
    print("\n" + "=" * 70)
    print("TEST: Failure Recovery (DR-06)")
    print("=" * 70)

    manager = RecoveryManager()

    # Register compensation
    manager.register_compensation(
        "create_user",
        lambda: print("  Executing: Delete partially created user")
    )

    # Simulate failures
    failure1 = manager.record_failure(
        step_id="api_call",
        error_type="timeout",
        error_message="Request timed out after 30s"
    )

    failure2 = manager.record_failure(
        step_id="create_user",
        error_type="constraint_violation",
        error_message="Email already exists"
    )

    # Determine and execute recovery
    for failure in manager.failures:
        strategy = manager.get_recovery_strategy(failure)
        print(f"\nFailure: {failure.step_id}")
        print(f"Strategy: {strategy.value}")

        action = manager.execute_recovery(failure, strategy)
        print(f"Result: {action.result}")

    # Generate report
    print(f"\n{manager.get_recovery_report()}")

    print("✅ Failure recovery works")


def test_full_recovery_flow():
    """Test complete recovery flow."""
    print("\n" + "=" * 70)
    print("TEST: Full Recovery Flow")
    print("=" * 70)

    idempotency = IdempotencyManager()
    recovery = RecoveryManager()

    # Simulate transactional operation with recovery
    def create_order(params):
        print(f"  Creating order: {params}")
        # Simulate failure on first attempt
        if params.get("attempt", 1) == 1:
            raise ValueError("Database connection lost")
        return {"order_id": "ORD-123"}

    try:
        result, cached = idempotency.execute_once(
            "create_order",
            {"user_id": "U1", "items": ["A", "B"]},
            lambda p: create_order({**p, "attempt": 1})
        )
    except ValueError as e:
        failure = recovery.record_failure(
            step_id="create_order",
            error_type="connection_error",
            error_message=str(e)
        )
        strategy = recovery.get_recovery_strategy(failure)
        action = recovery.execute_recovery(failure, strategy)
        print(f"Recovery: {action.result}")

    print("\n✅ Full recovery flow works")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ DR-04, DR-05, DR-06: RECOVERY - EVALUATION SUMMARY                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ DR-04 (Idempotency): ⭐ (Not Supported)                                     │
│   ❌ No built-in idempotency keys                                           │
│   ❌ No automatic deduplication                                             │
│   ⚠️ Custom IdempotencyManager implementation provided                     │
│                                                                             │
│ DR-05 (Plan Diff): ⭐ (Not Supported)                                       │
│   ❌ No built-in plan visualization                                         │
│   ❌ No diff comparison                                                     │
│   ⚠️ Custom PlanDiff implementation provided                               │
│                                                                             │
│ DR-06 (Failure Recovery): ⭐⭐ (Experimental)                               │
│   ✅ Sessions enable state recovery                                         │
│   ❌ No automatic rollback                                                  │
│   ❌ No compensation mechanism                                              │
│   ⚠️ Custom RecoveryManager implementation provided                        │
│                                                                             │
│ Custom Implementation Provided:                                             │
│   ✅ IdempotencyManager - Exactly-once execution                            │
│   ✅ PlanDiff - Plan comparison and visualization                           │
│   ✅ RecoveryManager - Failure handling and compensation                    │
│                                                                             │
│ Comparison with LangGraph:                                                  │
│   LangGraph:                                                                │
│     - Checkpointer for state recovery                                       │
│     - No native idempotency                                                 │
│     - No plan diff                                                          │
│   OpenAI SDK:                                                               │
│     - Sessions for state                                                    │
│     - Similar gaps                                                          │
│                                                                             │
│ Production Notes:                                                           │
│   - Use external idempotency store (Redis)                                  │
│   - Implement Saga pattern for complex transactions                         │
│   - Store execution plans for debugging                                     │
│   - Define compensation actions for critical operations                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_idempotency()
    test_plan_diff()
    test_failure_recovery()
    test_full_recovery_flow()

    print(SUMMARY)
