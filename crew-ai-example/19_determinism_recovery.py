"""
19_determinism_recovery.py - Determinism Recovery (DR-05, DR-06)

Purpose: Verify plan diff and failure recovery
- DR-05: Plan diff - present differences for approval
- DR-06: Failure recovery - recovery procedures after failure
- Partial apply/rollback support

LangGraph Comparison:
- Both require custom implementation for diff and recovery
- LangGraph checkpointer provides state recovery but not diff
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from enum import Enum
from copy import deepcopy
import difflib

from crewai.flow.flow import Flow, listen, start, router
from pydantic import BaseModel, Field


# =============================================================================
# Plan and Diff (DR-05)
# =============================================================================

class ChangeType(str, Enum):
    """Types of changes in a plan."""
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"
    NO_CHANGE = "no_change"


class PlanChange(BaseModel):
    """A single change in a plan."""

    id: str
    change_type: ChangeType
    resource: str
    field: Optional[str] = None
    before: Optional[Any] = None
    after: Optional[Any] = None
    description: str = ""
    risk_level: str = "low"  # low, medium, high


class ExecutionPlan(BaseModel):
    """An execution plan with changes to apply."""

    plan_id: str
    description: str
    changes: list[PlanChange] = []
    created_at: str = ""
    approved_at: Optional[str] = None
    approved_by: Optional[str] = None
    status: str = "pending"  # pending, approved, rejected, executing, completed, failed


class PlanDiff:
    """
    Generates and displays diffs between states (DR-05).

    Shows before/after changes for approval.
    """

    def __init__(self):
        self.change_counter = 0

    def _generate_id(self) -> str:
        """Generate unique change ID."""
        self.change_counter += 1
        return f"change_{self.change_counter:04d}"

    def compute_diff(self, before: dict, after: dict, resource: str = "root") -> list[PlanChange]:
        """
        Compute differences between two states.

        Returns list of changes.
        """
        changes = []

        # Find keys in after but not in before (additions)
        for key in after:
            if key not in before:
                changes.append(PlanChange(
                    id=self._generate_id(),
                    change_type=ChangeType.ADD,
                    resource=resource,
                    field=key,
                    before=None,
                    after=after[key],
                    description=f"Add {key} to {resource}",
                ))
            elif before[key] != after[key]:
                # Modification
                if isinstance(before[key], dict) and isinstance(after[key], dict):
                    # Recurse for nested dicts
                    nested_changes = self.compute_diff(
                        before[key], after[key], f"{resource}.{key}"
                    )
                    changes.extend(nested_changes)
                else:
                    changes.append(PlanChange(
                        id=self._generate_id(),
                        change_type=ChangeType.MODIFY,
                        resource=resource,
                        field=key,
                        before=before[key],
                        after=after[key],
                        description=f"Change {key} in {resource}",
                    ))

        # Find keys in before but not in after (deletions)
        for key in before:
            if key not in after:
                changes.append(PlanChange(
                    id=self._generate_id(),
                    change_type=ChangeType.DELETE,
                    resource=resource,
                    field=key,
                    before=before[key],
                    after=None,
                    description=f"Remove {key} from {resource}",
                ))

        return changes

    def format_diff(self, changes: list[PlanChange]) -> str:
        """Format changes as human-readable diff."""
        if not changes:
            return "No changes detected."

        lines = [
            "Plan Diff",
            "=" * 60,
            f"Total changes: {len(changes)}",
            "",
        ]

        for change in changes:
            symbol = {
                ChangeType.ADD: "+",
                ChangeType.MODIFY: "~",
                ChangeType.DELETE: "-",
            }.get(change.change_type, "?")

            lines.append(f"[{symbol}] {change.resource}.{change.field}")
            if change.change_type == ChangeType.ADD:
                lines.append(f"    + {change.after}")
            elif change.change_type == ChangeType.DELETE:
                lines.append(f"    - {change.before}")
            elif change.change_type == ChangeType.MODIFY:
                lines.append(f"    - {change.before}")
                lines.append(f"    + {change.after}")
            lines.append("")

        return "\n".join(lines)

    def format_text_diff(self, before: str, after: str) -> str:
        """Generate unified diff for text content."""
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)

        diff = difflib.unified_diff(
            before_lines, after_lines,
            fromfile='before', tofile='after'
        )

        return ''.join(diff)


# =============================================================================
# Recovery Strategies (DR-06)
# =============================================================================

class RecoveryStrategy(str, Enum):
    """Recovery strategies for failure handling."""
    RETRY = "retry"           # Retry the failed operation
    SKIP = "skip"             # Skip and continue
    ROLLBACK = "rollback"     # Rollback to previous state
    PARTIAL = "partial"       # Keep successful changes, report failures
    MANUAL = "manual"         # Require manual intervention
    COMPENSATE = "compensate" # Run compensating transactions


class RecoveryState(BaseModel):
    """State for recovery tracking."""

    operation_id: str
    status: str  # running, failed, recovering, recovered, rolled_back
    error_message: Optional[str] = None
    checkpoint_state: Optional[dict] = None
    applied_changes: list[str] = []  # IDs of successfully applied changes
    failed_changes: list[str] = []   # IDs of failed changes
    compensations_executed: list[str] = []
    recovery_attempts: int = 0


class RecoveryManager:
    """
    Manages failure recovery procedures (DR-06).

    Supports various recovery strategies including rollback
    and partial application.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./db/recovery")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.states: dict[str, RecoveryState] = {}
        self.checkpoints: dict[str, dict] = {}
        self.compensation_handlers: dict[str, callable] = {}

    def register_compensation(self, change_type: str, handler: callable):
        """Register a compensation handler for a change type."""
        self.compensation_handlers[change_type] = handler

    def create_checkpoint(self, operation_id: str, state: dict):
        """Create a checkpoint for potential rollback."""
        self.checkpoints[operation_id] = deepcopy(state)
        print(f"[Recovery] Checkpoint created: {operation_id}")

    def start_operation(self, operation_id: str, initial_state: dict) -> RecoveryState:
        """Start tracking an operation for recovery."""
        recovery_state = RecoveryState(
            operation_id=operation_id,
            status="running",
            checkpoint_state=deepcopy(initial_state),
        )
        self.states[operation_id] = recovery_state
        self.create_checkpoint(operation_id, initial_state)
        return recovery_state

    def record_success(self, operation_id: str, change_id: str):
        """Record a successful change."""
        if operation_id in self.states:
            self.states[operation_id].applied_changes.append(change_id)
            print(f"[Recovery] Change succeeded: {change_id}")

    def record_failure(self, operation_id: str, change_id: str, error: str):
        """Record a failed change."""
        if operation_id in self.states:
            state = self.states[operation_id]
            state.failed_changes.append(change_id)
            state.error_message = error
            state.status = "failed"
            print(f"[Recovery] Change failed: {change_id} - {error}")

    def recover(
        self,
        operation_id: str,
        strategy: RecoveryStrategy,
    ) -> dict:
        """
        Execute recovery based on strategy.

        Returns recovery result.
        """
        if operation_id not in self.states:
            return {"success": False, "error": "Operation not found"}

        state = self.states[operation_id]
        state.recovery_attempts += 1
        state.status = "recovering"

        print(f"[Recovery] Starting recovery for {operation_id}")
        print(f"[Recovery] Strategy: {strategy.value}")
        print(f"[Recovery] Applied changes: {len(state.applied_changes)}")
        print(f"[Recovery] Failed changes: {len(state.failed_changes)}")

        result = {"success": False, "strategy": strategy.value}

        if strategy == RecoveryStrategy.ROLLBACK:
            result = self._rollback(operation_id, state)

        elif strategy == RecoveryStrategy.PARTIAL:
            result = self._partial_recovery(operation_id, state)

        elif strategy == RecoveryStrategy.COMPENSATE:
            result = self._compensate(operation_id, state)

        elif strategy == RecoveryStrategy.RETRY:
            result = self._retry(operation_id, state)

        elif strategy == RecoveryStrategy.SKIP:
            state.status = "recovered"
            result = {"success": True, "message": "Skipped failed changes"}

        elif strategy == RecoveryStrategy.MANUAL:
            result = {"success": False, "message": "Manual intervention required"}

        # Save recovery state
        self._save_state(operation_id)
        return result

    def _rollback(self, operation_id: str, state: RecoveryState) -> dict:
        """Execute rollback to checkpoint state."""
        print(f"[Recovery] Rolling back to checkpoint...")

        if operation_id not in self.checkpoints:
            return {"success": False, "error": "No checkpoint found"}

        checkpoint = self.checkpoints[operation_id]

        # In real implementation, would restore actual state
        state.status = "rolled_back"
        return {
            "success": True,
            "message": f"Rolled back to checkpoint",
            "restored_state": checkpoint,
        }

    def _partial_recovery(self, operation_id: str, state: RecoveryState) -> dict:
        """Keep successful changes, report failures."""
        print(f"[Recovery] Partial recovery - keeping {len(state.applied_changes)} changes")

        state.status = "recovered"
        return {
            "success": True,
            "message": "Partial recovery completed",
            "applied": state.applied_changes,
            "failed": state.failed_changes,
        }

    def _compensate(self, operation_id: str, state: RecoveryState) -> dict:
        """Execute compensating transactions for applied changes."""
        print(f"[Recovery] Executing compensating transactions...")

        for change_id in reversed(state.applied_changes):
            # Look up compensation handler
            # In real implementation, would execute actual compensation
            print(f"[Recovery] Compensating: {change_id}")
            state.compensations_executed.append(change_id)

        state.status = "recovered"
        return {
            "success": True,
            "message": f"Compensated {len(state.compensations_executed)} changes",
        }

    def _retry(self, operation_id: str, state: RecoveryState) -> dict:
        """Retry failed changes."""
        print(f"[Recovery] Retrying {len(state.failed_changes)} failed changes...")

        # In real implementation, would re-execute failed changes
        # For demo, just clear failed list
        retried = state.failed_changes.copy()
        state.failed_changes = []
        state.applied_changes.extend(retried)
        state.status = "recovered"

        return {
            "success": True,
            "message": f"Retried {len(retried)} changes",
        }

    def _save_state(self, operation_id: str):
        """Save recovery state to file."""
        if operation_id not in self.states:
            return

        state = self.states[operation_id]
        output_file = self.storage_path / f"{operation_id}.json"

        with open(output_file, "w") as f:
            json.dump(state.model_dump(), f, indent=2, default=str)

    def get_recovery_report(self, operation_id: str) -> str:
        """Generate recovery report."""
        if operation_id not in self.states:
            return "Operation not found"

        state = self.states[operation_id]

        return f"""
Recovery Report: {operation_id}
{'=' * 60}
Status: {state.status}
Recovery Attempts: {state.recovery_attempts}
Applied Changes: {len(state.applied_changes)}
Failed Changes: {len(state.failed_changes)}
Compensations: {len(state.compensations_executed)}
Error: {state.error_message or 'None'}
"""


# =============================================================================
# Recovery Flow State
# =============================================================================

class RecoverableState(BaseModel):
    """State for recoverable workflow."""

    workflow_id: str = ""
    current_step: str = "init"
    data: dict = {}
    changes_to_apply: list[dict] = []
    applied_changes: list[str] = []
    failed_changes: list[str] = []
    recovery_strategy: str = "partial"
    is_recovered: bool = False


# =============================================================================
# Recoverable Workflow
# =============================================================================

class RecoverableWorkflow(Flow[RecoverableState]):
    """
    Workflow with failure recovery capabilities.

    Demonstrates plan diff and recovery procedures.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plan_diff = PlanDiff()
        self.recovery_manager = RecoveryManager()

    @start()
    def initialize(self):
        """Initialize workflow."""
        print("\n[Initialize] Starting recoverable workflow...")

        self.state.workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initial state
        self.state.data = {
            "users": {"count": 100},
            "orders": {"pending": 5, "completed": 95},
            "inventory": {"items": 1000},
        }

    @listen(initialize)
    def create_plan(self):
        """Create execution plan with changes."""
        print("\n[Plan] Creating execution plan...")

        # Define changes to apply
        new_state = {
            "users": {"count": 110},  # Modified
            "orders": {"pending": 3, "completed": 97, "cancelled": 5},  # Modified + Added
            "inventory": {"items": 950},  # Modified
            "metrics": {"daily_active": 50},  # New
        }

        # Compute diff
        changes = self.plan_diff.compute_diff(self.state.data, new_state)

        self.state.changes_to_apply = [c.model_dump() for c in changes]

        # Display diff for approval
        print("\n" + self.plan_diff.format_diff(changes))

    @listen(create_plan)
    def request_approval(self):
        """Request approval for the plan."""
        print("\n" + "=" * 60)
        print("PLAN APPROVAL REQUIRED")
        print("=" * 60)
        print(f"Total changes: {len(self.state.changes_to_apply)}")
        print("\nOptions:")
        print("  1. approve - Apply all changes")
        print("  2. reject - Cancel execution")
        print("  3. partial - Apply safe changes only")

        # Auto-approve for demo
        print("\n[Auto-approved for demo]")
        return "approved"

    @router(request_approval)
    def route_after_approval(self, approval: str):
        """Route based on approval."""
        if approval == "approved":
            return "execute"
        elif approval == "partial":
            return "execute_partial"
        else:
            return "cancel"

    @listen("execute")
    def execute_changes(self):
        """Execute the approved changes."""
        print("\n[Execute] Applying changes...")

        operation_id = self.state.workflow_id
        self.recovery_manager.start_operation(operation_id, self.state.data)

        # Simulate applying changes with some failures
        for i, change in enumerate(self.state.changes_to_apply):
            change_id = change.get("id", f"change_{i}")

            # Simulate failure on 3rd change
            if i == 2:
                self.recovery_manager.record_failure(
                    operation_id, change_id, "Simulated failure"
                )
                self.state.failed_changes.append(change_id)
            else:
                self.recovery_manager.record_success(operation_id, change_id)
                self.state.applied_changes.append(change_id)
                print(f"[Execute] Applied: {change_id}")

        if self.state.failed_changes:
            print(f"\n[Execute] {len(self.state.failed_changes)} changes failed!")
            return "recover"
        else:
            return "complete"

    @router(execute_changes)
    def route_after_execute(self, result: str):
        """Route based on execution result."""
        return result

    @listen("recover")
    def recover_from_failure(self):
        """Recover from execution failure."""
        print("\n[Recover] Initiating recovery...")

        strategy = RecoveryStrategy(self.state.recovery_strategy)
        result = self.recovery_manager.recover(
            self.state.workflow_id,
            strategy,
        )

        print(f"[Recover] Result: {result}")
        self.state.is_recovered = result.get("success", False)

    @listen(recover_from_failure)
    def report_recovery(self):
        """Report recovery status."""
        report = self.recovery_manager.get_recovery_report(self.state.workflow_id)
        print(report)
        return report

    @listen("execute_partial")
    def execute_partial(self):
        """Execute only safe changes."""
        print("\n[Execute] Applying safe changes only...")
        # Would filter to only low-risk changes
        return "complete"

    @listen("cancel")
    def cancel_execution(self):
        """Cancel the execution."""
        print("\n[Cancel] Execution cancelled.")
        return "Execution cancelled by user."

    @listen("complete")
    def complete_workflow(self):
        """Complete the workflow."""
        print("\n[Complete] Workflow completed successfully!")
        return f"""
Workflow Complete
=================
ID: {self.state.workflow_id}
Applied: {len(self.state.applied_changes)} changes
Failed: {len(self.state.failed_changes)} changes
Recovered: {self.state.is_recovered}
"""


# =============================================================================
# Demonstrations
# =============================================================================

def demo_plan_diff():
    """Demonstrate plan diff generation."""
    print("=" * 60)
    print("Demo 1: Plan Diff (DR-05)")
    print("=" * 60)

    differ = PlanDiff()

    before = {
        "config": {
            "timeout": 30,
            "retries": 3,
        },
        "features": {
            "feature_a": True,
            "feature_b": False,
        },
        "deprecated": "old_value",
    }

    after = {
        "config": {
            "timeout": 60,  # Modified
            "retries": 3,
            "batch_size": 100,  # Added
        },
        "features": {
            "feature_a": True,
            "feature_b": True,  # Modified
            "feature_c": True,  # Added
        },
        # deprecated removed
    }

    changes = differ.compute_diff(before, after)
    print("\n" + differ.format_diff(changes))


def demo_text_diff():
    """Demonstrate text-based diff."""
    print("\n" + "=" * 60)
    print("Demo 2: Text Diff")
    print("=" * 60)

    differ = PlanDiff()

    before_text = """def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
"""

    after_text = """def process_data(data, multiplier=2):
    result = []
    for item in data:
        if item is not None:
            result.append(item * multiplier)
    return result
"""

    diff = differ.format_text_diff(before_text, after_text)
    print("\nCode diff:")
    print(diff)


def demo_recovery_strategies():
    """Demonstrate various recovery strategies."""
    print("\n" + "=" * 60)
    print("Demo 3: Recovery Strategies (DR-06)")
    print("=" * 60)

    manager = RecoveryManager()

    # Simulate an operation with failures
    op_id = "test_operation"
    initial_state = {"data": [1, 2, 3]}

    manager.start_operation(op_id, initial_state)

    # Record some successes and failures
    manager.record_success(op_id, "change_1")
    manager.record_success(op_id, "change_2")
    manager.record_failure(op_id, "change_3", "Database connection failed")

    print("\nTesting different recovery strategies:\n")

    # Test each strategy
    strategies = [
        RecoveryStrategy.PARTIAL,
        RecoveryStrategy.ROLLBACK,
        RecoveryStrategy.RETRY,
        RecoveryStrategy.SKIP,
    ]

    for strategy in strategies:
        # Reset state for each test
        manager.states[op_id].status = "failed"
        manager.states[op_id].recovery_attempts = 0

        print(f"\n--- Strategy: {strategy.value} ---")
        result = manager.recover(op_id, strategy)
        print(f"Result: {result}")


def demo_recoverable_workflow():
    """Demonstrate recoverable workflow."""
    print("\n" + "=" * 60)
    print("Demo 4: Recoverable Workflow")
    print("=" * 60)

    flow = RecoverableWorkflow()
    result = flow.kickoff()
    print(result)


def demo_checkpoint_restore():
    """Demonstrate checkpoint and restore."""
    print("\n" + "=" * 60)
    print("Demo 5: Checkpoint and Restore")
    print("=" * 60)

    manager = RecoveryManager()

    # Create checkpoint
    op_id = "checkpoint_test"
    original_state = {
        "balance": 1000,
        "transactions": [],
    }

    manager.create_checkpoint(op_id, original_state)
    print(f"Original state: {original_state}")

    # Simulate state changes
    modified_state = {
        "balance": 500,
        "transactions": ["tx1", "tx2"],
    }
    print(f"Modified state: {modified_state}")

    # Restore from checkpoint
    if op_id in manager.checkpoints:
        restored = manager.checkpoints[op_id]
        print(f"Restored state: {restored}")
        print(f"Match original: {restored == original_state}")


def main():
    print("=" * 60)
    print("Determinism Recovery Verification (DR-05, DR-06)")
    print("=" * 60)
    print("""
This script verifies plan diff and failure recovery capabilities.

Verification Items:
- DR-05: Plan Diff
  - Before/after state comparison
  - Change detection and formatting
  - Approval workflow for changes

- DR-06: Failure Recovery
  - Checkpoint creation
  - Multiple recovery strategies
  - Partial application support
  - Rollback capability

Recovery Strategies:
- RETRY: Re-attempt failed operations
- SKIP: Continue without failed changes
- ROLLBACK: Restore to checkpoint state
- PARTIAL: Keep successful, report failures
- COMPENSATE: Undo applied changes
- MANUAL: Require human intervention

LangGraph Comparison:
- LangGraph checkpointer provides state recovery
- Neither has built-in diff visualization
- Both require custom recovery logic
""")

    # Run all demos
    demo_plan_diff()
    demo_text_diff()
    demo_recovery_strategies()
    demo_recoverable_workflow()
    demo_checkpoint_restore()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
