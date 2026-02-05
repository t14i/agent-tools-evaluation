"""
LangGraph Determinism - Plan Diff & Failure Recovery (DR-05, DR-06)
Visualizing changes before execution and handling failures.

Evaluation: DR-05 (Plan Diff), DR-06 (Failure Recovery)
"""

import json
from typing import Annotated, TypedDict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


# =============================================================================
# PLAN DIFF SYSTEM (DR-05)
# =============================================================================

class ChangeType(Enum):
    """Types of changes."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    NO_CHANGE = "no_change"


@dataclass
class PlannedChange:
    """A single planned change."""
    resource_type: str  # "file", "database", "email", etc.
    resource_id: str
    change_type: ChangeType
    current_state: Optional[Any] = None
    planned_state: Optional[Any] = None
    reversible: bool = True
    risk_level: str = "low"  # low, medium, high, critical


@dataclass
class ExecutionPlan:
    """Complete execution plan with all changes."""
    plan_id: str
    created_at: datetime
    description: str
    changes: list[PlannedChange]
    dependencies: dict[str, list[str]] = field(default_factory=dict)  # change_id -> [depends_on]
    approved: bool = False
    executed: bool = False


class PlanDiffGenerator:
    """
    Generates plan diffs for review before execution.
    Implements DR-05: Plan Diff.
    """

    def __init__(self):
        self.current_state: dict[str, Any] = {}  # Simulated current state

    def set_current_state(self, resource_type: str, resource_id: str, state: Any):
        """Set current state for a resource."""
        key = f"{resource_type}:{resource_id}"
        self.current_state[key] = state

    def generate_plan(
        self,
        description: str,
        tool_calls: list[dict]
    ) -> ExecutionPlan:
        """Generate a plan from tool calls."""
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        changes = []

        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            change = self._tool_to_change(tool_name, tool_args)
            if change:
                changes.append(change)

        return ExecutionPlan(
            plan_id=plan_id,
            created_at=datetime.now(),
            description=description,
            changes=changes
        )

    def _tool_to_change(self, tool_name: str, tool_args: dict) -> Optional[PlannedChange]:
        """Convert a tool call to a planned change."""
        if tool_name == "create_file":
            return PlannedChange(
                resource_type="file",
                resource_id=tool_args.get("path", "unknown"),
                change_type=ChangeType.CREATE,
                current_state=None,
                planned_state={"content": tool_args.get("content", "")[:100]},
                reversible=True,
                risk_level="low"
            )
        elif tool_name == "update_file":
            path = tool_args.get("path", "unknown")
            key = f"file:{path}"
            return PlannedChange(
                resource_type="file",
                resource_id=path,
                change_type=ChangeType.UPDATE,
                current_state=self.current_state.get(key),
                planned_state={"content": tool_args.get("content", "")[:100]},
                reversible=True,
                risk_level="medium"
            )
        elif tool_name == "delete_file":
            path = tool_args.get("path", "unknown")
            key = f"file:{path}"
            return PlannedChange(
                resource_type="file",
                resource_id=path,
                change_type=ChangeType.DELETE,
                current_state=self.current_state.get(key),
                planned_state=None,
                reversible=False,
                risk_level="high"
            )
        elif tool_name == "update_database":
            table = tool_args.get("table", "unknown")
            return PlannedChange(
                resource_type="database",
                resource_id=table,
                change_type=ChangeType.UPDATE,
                current_state="[current rows]",
                planned_state=tool_args.get("data"),
                reversible=True,
                risk_level="high"
            )
        elif tool_name == "send_email":
            return PlannedChange(
                resource_type="email",
                resource_id=tool_args.get("to", "unknown"),
                change_type=ChangeType.CREATE,
                current_state=None,
                planned_state={"subject": tool_args.get("subject"), "body": tool_args.get("body", "")[:50]},
                reversible=False,
                risk_level="medium"
            )
        return None

    def format_diff(self, plan: ExecutionPlan) -> str:
        """Format plan as human-readable diff."""
        lines = [
            f"═══════════════════════════════════════════════════════════════",
            f"EXECUTION PLAN: {plan.plan_id}",
            f"Description: {plan.description}",
            f"Created: {plan.created_at.isoformat()}",
            f"Changes: {len(plan.changes)}",
            f"═══════════════════════════════════════════════════════════════"
        ]

        for i, change in enumerate(plan.changes, 1):
            symbol = {
                ChangeType.CREATE: "+",
                ChangeType.UPDATE: "~",
                ChangeType.DELETE: "-",
                ChangeType.NO_CHANGE: " "
            }[change.change_type]

            risk_emoji = {
                "low": "🟢",
                "medium": "🟡",
                "high": "🟠",
                "critical": "🔴"
            }[change.risk_level]

            lines.append(f"\n[{i}] {symbol} {change.resource_type.upper()}: {change.resource_id}")
            lines.append(f"    Risk: {risk_emoji} {change.risk_level} | Reversible: {'Yes' if change.reversible else 'No'}")

            if change.current_state is not None:
                lines.append(f"    Current: {str(change.current_state)[:60]}...")
            if change.planned_state is not None:
                lines.append(f"    Planned: {str(change.planned_state)[:60]}...")

        return "\n".join(lines)


# =============================================================================
# FAILURE RECOVERY SYSTEM (DR-06)
# =============================================================================

class RecoveryStrategy(Enum):
    """Recovery strategies."""
    RETRY = "retry"  # Retry the failed operation
    SKIP = "skip"  # Skip and continue
    ROLLBACK = "rollback"  # Rollback completed operations
    COMPENSATE = "compensate"  # Run compensating actions
    ABORT = "abort"  # Stop execution entirely


@dataclass
class CompletedOperation:
    """Record of a completed operation for rollback."""
    operation_id: str
    tool_name: str
    tool_args: dict
    result: Any
    timestamp: datetime
    reversible: bool
    reverse_operation: Optional[dict] = None  # Tool call to reverse this


@dataclass
class FailureRecord:
    """Record of a failure."""
    operation_id: str
    tool_name: str
    tool_args: dict
    error: str
    timestamp: datetime
    recovery_strategy: Optional[RecoveryStrategy] = None


class RecoveryManager:
    """
    Manages failure recovery and rollback.
    Implements DR-06: Failure Recovery.
    """

    def __init__(self):
        self.completed_operations: list[CompletedOperation] = []
        self.failures: list[FailureRecord] = []
        self.compensating_actions: list[dict] = []

    def record_success(
        self,
        tool_name: str,
        tool_args: dict,
        result: Any,
        reverse_operation: Optional[dict] = None
    ):
        """Record a successful operation."""
        op = CompletedOperation(
            operation_id=f"op_{len(self.completed_operations)}",
            tool_name=tool_name,
            tool_args=tool_args,
            result=result,
            timestamp=datetime.now(),
            reversible=reverse_operation is not None,
            reverse_operation=reverse_operation
        )
        self.completed_operations.append(op)
        print(f"  [SUCCESS] Recorded: {tool_name}")

    def record_failure(
        self,
        tool_name: str,
        tool_args: dict,
        error: str
    ) -> FailureRecord:
        """Record a failure."""
        failure = FailureRecord(
            operation_id=f"fail_{len(self.failures)}",
            tool_name=tool_name,
            tool_args=tool_args,
            error=error,
            timestamp=datetime.now()
        )
        self.failures.append(failure)
        print(f"  [FAILURE] Recorded: {tool_name} - {error}")
        return failure

    def generate_rollback_plan(self) -> list[dict]:
        """Generate plan to rollback completed operations."""
        rollback_ops = []
        for op in reversed(self.completed_operations):
            if op.reversible and op.reverse_operation:
                rollback_ops.append({
                    "original_operation": op.operation_id,
                    "reverse": op.reverse_operation
                })
        return rollback_ops

    def execute_rollback(self, tools_by_name: dict) -> list[tuple[str, Any]]:
        """Execute rollback operations."""
        results = []
        rollback_plan = self.generate_rollback_plan()

        print(f"\n  [ROLLBACK] Executing {len(rollback_plan)} rollback operations")
        for rollback in rollback_plan:
            reverse = rollback["reverse"]
            tool_name = reverse["name"]
            tool_args = reverse["args"]

            try:
                if tool_name in tools_by_name:
                    result = tools_by_name[tool_name].invoke(tool_args)
                    results.append((tool_name, result))
                    print(f"    ✓ Rolled back: {rollback['original_operation']}")
            except Exception as e:
                results.append((tool_name, f"Error: {e}"))
                print(f"    ✗ Rollback failed: {tool_name} - {e}")

        return results

    def suggest_recovery(self, failure: FailureRecord) -> RecoveryStrategy:
        """Suggest recovery strategy based on failure type."""
        error = failure.error.lower()

        if "timeout" in error or "connection" in error:
            return RecoveryStrategy.RETRY
        elif "not found" in error:
            return RecoveryStrategy.SKIP
        elif "permission" in error or "denied" in error:
            return RecoveryStrategy.ABORT
        elif "conflict" in error:
            return RecoveryStrategy.COMPENSATE
        else:
            return RecoveryStrategy.ROLLBACK

    def clear(self):
        """Clear all records."""
        self.completed_operations.clear()
        self.failures.clear()
        self.compensating_actions.clear()


# =============================================================================
# STATE AND TOOLS
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    execution_plan: Optional[dict]
    recovery_mode: bool


# Initialize systems
plan_generator = PlanDiffGenerator()
recovery_manager = RecoveryManager()


@tool
def create_file(path: str, content: str) -> str:
    """Create a new file."""
    return f"Created file: {path} ({len(content)} bytes)"


@tool
def update_file(path: str, content: str) -> str:
    """Update an existing file."""
    return f"Updated file: {path} ({len(content)} bytes)"


@tool
def delete_file(path: str) -> str:
    """Delete a file (irreversible)."""
    return f"Deleted file: {path}"


@tool
def update_database(table: str, data: str) -> str:
    """Update database table."""
    import random
    if random.random() < 0.3:  # Simulate occasional failure
        raise Exception("Database connection timeout")
    return f"Updated table {table} with data"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (irreversible)."""
    return f"Sent email to {to}: {subject}"


tools = [create_file, update_file, delete_file, update_database, send_email]

# Define reverse operations
REVERSE_OPERATIONS = {
    "create_file": lambda args: {"name": "delete_file", "args": {"path": args["path"]}},
    "update_file": lambda args: None,  # Would need original content
    "delete_file": lambda args: None,  # Irreversible
    "send_email": lambda args: None,  # Irreversible
}


# =============================================================================
# GRAPH NODES
# =============================================================================

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


def agent(state: State) -> State:
    """Agent node."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def plan_review(state: State) -> Command:
    """
    Plan review node - generates diff and requests approval.
    Implements DR-05: Plan Diff.
    """
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return Command(goto="tools")

    # Generate execution plan
    plan = plan_generator.generate_plan(
        description="Agent-proposed changes",
        tool_calls=last_message.tool_calls
    )

    # Format and display diff
    diff = plan_generator.format_diff(plan)
    print(f"\n{diff}")

    # Check for high-risk changes
    high_risk_changes = [c for c in plan.changes if c.risk_level in ["high", "critical"]]
    irreversible_changes = [c for c in plan.changes if not c.reversible]

    if high_risk_changes or irreversible_changes:
        # Interrupt for approval
        decision = interrupt({
            "action": "approve_plan",
            "plan_id": plan.plan_id,
            "total_changes": len(plan.changes),
            "high_risk_count": len(high_risk_changes),
            "irreversible_count": len(irreversible_changes),
            "diff_preview": diff[:500]
        })

        if not decision.get("approved"):
            rejection_msg = ToolMessage(
                content=f"Plan rejected: {decision.get('reason', 'No reason provided')}",
                tool_call_id=last_message.tool_calls[0]["id"]
            )
            return Command(goto="agent", update={"messages": [rejection_msg]})

    return Command(goto="tools", update={"execution_plan": {"plan_id": plan.plan_id}})


class RecoveryToolNode:
    """
    Tool node with failure recovery.
    Implements DR-06: Failure Recovery.
    """

    def __init__(self, tools: list, recovery_manager: RecoveryManager):
        self.tool_node = ToolNode(tools)
        self.tools_by_name = {t.name: t for t in tools}
        self.recovery_manager = recovery_manager

    def __call__(self, state: dict) -> dict:
        """Execute tools with recovery handling."""
        last_message = state["messages"][-1]
        results = []
        recovery_mode = state.get("recovery_mode", False)

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            try:
                # Execute tool
                tool_fn = self.tools_by_name[tool_name]
                result = tool_fn.invoke(tool_args)

                # Record success with reverse operation
                reverse_fn = REVERSE_OPERATIONS.get(tool_name)
                reverse_op = reverse_fn(tool_args) if reverse_fn else None
                self.recovery_manager.record_success(tool_name, tool_args, result, reverse_op)

                results.append(ToolMessage(content=str(result), tool_call_id=tool_id))

            except Exception as e:
                # Record failure
                failure = self.recovery_manager.record_failure(tool_name, tool_args, str(e))

                # Suggest recovery
                suggested_strategy = self.recovery_manager.suggest_recovery(failure)
                print(f"  [RECOVERY] Suggested strategy: {suggested_strategy.value}")

                if recovery_mode:
                    # Automatic recovery based on strategy
                    if suggested_strategy == RecoveryStrategy.RETRY:
                        # Retry once
                        try:
                            result = self.tools_by_name[tool_name].invoke(tool_args)
                            results.append(ToolMessage(content=f"[RETRY SUCCESS] {result}", tool_call_id=tool_id))
                            continue
                        except Exception:
                            pass

                    elif suggested_strategy == RecoveryStrategy.ROLLBACK:
                        # Execute rollback
                        rollback_results = self.recovery_manager.execute_rollback(self.tools_by_name)
                        results.append(ToolMessage(
                            content=f"[ROLLBACK] Rolled back {len(rollback_results)} operations due to: {e}",
                            tool_call_id=tool_id
                        ))
                        continue

                # Default: return error
                results.append(ToolMessage(
                    content=f"Error: {e} (Suggested: {suggested_strategy.value})",
                    tool_call_id=tool_id
                ))

        return {"messages": results}


def should_continue(state: State) -> str:
    """Conditional edge."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "plan_review"
    return END


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_graph():
    """Build graph with plan diff and recovery."""
    builder = StateGraph(State)

    builder.add_node("agent", agent)
    builder.add_node("plan_review", plan_review)
    builder.add_node("tools", RecoveryToolNode(tools, recovery_manager))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["plan_review", END])
    builder.add_edge("plan_review", "tools")
    builder.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# TESTS
# =============================================================================

def test_plan_diff_generation():
    """Test plan diff generation (DR-05)."""
    print("\n" + "=" * 70)
    print("TEST: Plan Diff Generation (DR-05)")
    print("=" * 70)

    # Set some current state
    plan_generator.set_current_state("file", "/data/config.json", {"content": "old config"})

    # Generate plan from tool calls
    tool_calls = [
        {"name": "create_file", "args": {"path": "/data/new.txt", "content": "new content"}},
        {"name": "update_file", "args": {"path": "/data/config.json", "content": "new config"}},
        {"name": "delete_file", "args": {"path": "/data/old.txt"}},
        {"name": "send_email", "args": {"to": "user@example.com", "subject": "Update", "body": "Changes made"}},
    ]

    plan = plan_generator.generate_plan("Test operation plan", tool_calls)
    diff = plan_generator.format_diff(plan)
    print(diff)


def test_failure_recovery():
    """Test failure recovery (DR-06)."""
    print("\n" + "=" * 70)
    print("TEST: Failure Recovery (DR-06)")
    print("=" * 70)

    # Clear previous records
    recovery_manager.clear()

    # Simulate successful operations
    recovery_manager.record_success(
        "create_file",
        {"path": "/tmp/test1.txt", "content": "content1"},
        "Created file",
        {"name": "delete_file", "args": {"path": "/tmp/test1.txt"}}
    )
    recovery_manager.record_success(
        "create_file",
        {"path": "/tmp/test2.txt", "content": "content2"},
        "Created file",
        {"name": "delete_file", "args": {"path": "/tmp/test2.txt"}}
    )

    # Simulate failure
    failure = recovery_manager.record_failure(
        "update_database",
        {"table": "users", "data": "..."},
        "Database connection timeout"
    )

    # Check suggested recovery
    strategy = recovery_manager.suggest_recovery(failure)
    print(f"\nSuggested recovery: {strategy.value}")

    # Generate rollback plan
    rollback_plan = recovery_manager.generate_rollback_plan()
    print(f"\nRollback plan ({len(rollback_plan)} operations):")
    for op in rollback_plan:
        print(f"  - Reverse {op['original_operation']}: {op['reverse']}")


def test_recovery_strategies():
    """Test different recovery strategy suggestions."""
    print("\n" + "=" * 70)
    print("TEST: Recovery Strategy Selection")
    print("=" * 70)

    test_cases = [
        ("Connection timeout error", "update_database", {}),
        ("Resource not found", "read_file", {}),
        ("Permission denied", "delete_file", {}),
        ("Version conflict detected", "update_database", {}),
        ("Unknown error occurred", "send_email", {}),
    ]

    for error, tool_name, args in test_cases:
        failure = FailureRecord(
            operation_id="test",
            tool_name=tool_name,
            tool_args=args,
            error=error,
            timestamp=datetime.now()
        )
        strategy = recovery_manager.suggest_recovery(failure)
        print(f"  Error: '{error}' → Strategy: {strategy.value}")


def test_integrated_plan_review():
    """Test integrated plan review with approval."""
    print("\n" + "=" * 70)
    print("TEST: Integrated Plan Review")
    print("=" * 70)

    graph = build_graph()
    config = {"configurable": {"thread_id": "plan-test-1"}}

    # This should trigger plan review
    result = graph.invoke(
        {
            "messages": [("user", "Create a file at /data/test.txt with content 'hello' and delete /data/old.txt")],
            "execution_plan": None,
            "recovery_mode": False
        },
        config=config
    )

    state = graph.get_state(config)
    if state.next:
        print(f"\nPlan requires approval. Interrupted at: {state.next}")
        print("Interrupt value available for review.")
    else:
        print(f"\nExecution completed: {result['messages'][-1].content[:100]}...")


# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ DR-05 & DR-06: PLAN DIFF & RECOVERY - EVALUATION SUMMARY                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LangGraph Native Support: ⭐ (Not Supported)                                │
│                                                                             │
│ LangGraph does NOT provide:                                                 │
│   ❌ Plan diff generation                                                   │
│   ❌ Change visualization                                                   │
│   ❌ Rollback mechanism                                                     │
│   ❌ Compensating transactions                                              │
│   ❌ Recovery strategies                                                    │
│                                                                             │
│ Custom Implementation Required:                                             │
│   ✅ PlannedChange - Model individual changes                               │
│   ✅ ExecutionPlan - Group changes with dependencies                        │
│   ✅ PlanDiffGenerator - Generate human-readable diffs                      │
│   ✅ RecoveryManager - Track operations for rollback                        │
│   ✅ RecoveryToolNode - Tool wrapper with failure handling                  │
│                                                                             │
│ DR-05 (Plan Diff) Features:                                                 │
│   ✓ Convert tool calls to planned changes                                   │
│   ✓ Show current vs planned state                                           │
│   ✓ Risk level indicators                                                   │
│   ✓ Reversibility marking                                                   │
│   ✓ Human-readable diff format                                              │
│                                                                             │
│ DR-06 (Failure Recovery) Features:                                          │
│   ✓ Track completed operations with reverse                                 │
│   ✓ Record failures with context                                            │
│   ✓ Suggest recovery strategies                                             │
│   ✓ Generate rollback plans                                                 │
│   ✓ Execute rollback operations                                             │
│                                                                             │
│ Recovery Strategies Supported:                                              │
│   - RETRY: For transient failures                                           │
│   - SKIP: For non-critical failures                                         │
│   - ROLLBACK: Undo completed operations                                     │
│   - COMPENSATE: Run compensating actions                                    │
│   - ABORT: Stop execution entirely                                          │
│                                                                             │
│ Production Considerations:                                                  │
│   - Persistent operation log                                                │
│   - Transactional boundaries                                                │
│   - Distributed transaction support                                         │
│   - Saga pattern for long-running operations                                │
│   - Point-in-time recovery                                                  │
│                                                                             │
│ Rating:                                                                     │
│   DR-05 (Plan Diff): ⭐ (fully custom)                                      │
│   DR-06 (Recovery): ⭐ (fully custom)                                       │
│                                                                             │
│   Essential for production safety but requires significant implementation.  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_plan_diff_generation()
    test_failure_recovery()
    test_recovery_strategies()
    test_integrated_plan_review()

    print(SUMMARY)
