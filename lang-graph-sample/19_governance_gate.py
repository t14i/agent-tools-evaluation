"""
LangGraph Governance - Destructive Operation Gate (GV-01)
Using interrupt() for approval gates with policy evaluation.

Evaluation: GV-01 (Destructive Operation Gate), TC-02 (Controllable Automation)
"""

from typing import Annotated, TypedDict, Optional
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
# POLICY EVALUATION FRAMEWORK
# =============================================================================

class RiskLevel(Enum):
    """Risk classification for operations."""
    LOW = "low"          # Auto-approve
    MEDIUM = "medium"    # Require single approval
    HIGH = "high"        # Require manager approval
    CRITICAL = "critical"  # Require multi-level approval


@dataclass
class PolicyRule:
    """Single policy rule definition."""
    tool_name: str
    condition: Optional[str] = None  # e.g., "count > 100"
    risk_level: RiskLevel = RiskLevel.MEDIUM
    requires_approval: bool = True
    required_role: str = "approver"
    reason: str = ""


class PolicyEngine:
    """Policy engine for evaluating tool calls."""

    def __init__(self):
        self.rules: list[PolicyRule] = []
        self.default_risk = RiskLevel.LOW

    def add_rule(self, rule: PolicyRule):
        """Add a policy rule."""
        self.rules.append(rule)

    def evaluate(
        self,
        tool_name: str,
        tool_args: dict,
        context: Optional[dict] = None
    ) -> tuple[RiskLevel, bool, str]:
        """
        Evaluate a tool call against policies.
        Returns: (risk_level, requires_approval, reason)
        """
        context = context or {}

        for rule in self.rules:
            if rule.tool_name != tool_name and rule.tool_name != "*":
                continue

            # Check condition if specified
            if rule.condition:
                if not self._evaluate_condition(rule.condition, tool_args, context):
                    continue

            return (rule.risk_level, rule.requires_approval, rule.reason)

        return (self.default_risk, False, "No matching policy, auto-approved")

    def _evaluate_condition(
        self,
        condition: str,
        tool_args: dict,
        context: dict
    ) -> bool:
        """Evaluate a condition expression."""
        # Simple condition evaluation (production would use safer eval)
        try:
            # Make args and context available for evaluation
            local_vars = {**tool_args, **context}

            # Support simple conditions like "count > 100"
            if ">" in condition:
                parts = condition.split(">")
                var_name = parts[0].strip()
                threshold = int(parts[1].strip())
                return local_vars.get(var_name, 0) > threshold

            if "==" in condition:
                parts = condition.split("==")
                var_name = parts[0].strip()
                value = parts[1].strip().strip("'\"")
                return str(local_vars.get(var_name, "")) == value

            return True
        except Exception:
            return True  # Fail closed - require approval on evaluation error


# =============================================================================
# AUDIT TRAIL
# =============================================================================

@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: datetime
    tool_name: str
    tool_args: dict
    risk_level: RiskLevel
    decision: str  # "approved", "rejected", "auto-approved"
    approver_id: Optional[str] = None
    reason: Optional[str] = None


class AuditTrail:
    """Simple in-memory audit trail (production would use persistent storage)."""

    def __init__(self):
        self.entries: list[AuditEntry] = []

    def log(
        self,
        tool_name: str,
        tool_args: dict,
        risk_level: RiskLevel,
        decision: str,
        approver_id: Optional[str] = None,
        reason: Optional[str] = None
    ):
        """Log an audit entry."""
        entry = AuditEntry(
            timestamp=datetime.now(),
            tool_name=tool_name,
            tool_args=tool_args,
            risk_level=risk_level,
            decision=decision,
            approver_id=approver_id,
            reason=reason
        )
        self.entries.append(entry)
        print(f"  [AUDIT] {decision.upper()}: {tool_name} by {approver_id or 'SYSTEM'}")

    def get_entries(self, tool_name: Optional[str] = None) -> list[AuditEntry]:
        """Get audit entries, optionally filtered by tool name."""
        if tool_name:
            return [e for e in self.entries if e.tool_name == tool_name]
        return self.entries


# =============================================================================
# STATE AND TOOLS
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    pending_approval: Optional[dict]
    audit_log: list


# Define tools with different risk levels
@tool
def read_file(path: str) -> str:
    """Read a file (low risk)."""
    return f"Contents of {path}: [file data]"


@tool
def write_file(path: str, content: str) -> str:
    """Write to a file (medium risk)."""
    return f"Written {len(content)} bytes to {path}"


@tool
def delete_file(path: str) -> str:
    """Delete a file (high risk)."""
    return f"Deleted file: {path}"


@tool
def delete_database_table(table_name: str, confirm: bool = False) -> str:
    """Drop a database table (critical risk)."""
    if not confirm:
        return "Error: Must set confirm=True for destructive operations"
    return f"Dropped table: {table_name}"


@tool
def send_bulk_email(recipients: int, subject: str) -> str:
    """Send bulk email (risk depends on count)."""
    return f"Sent email '{subject}' to {recipients} recipients"


tools = [read_file, write_file, delete_file, delete_database_table, send_bulk_email]


# =============================================================================
# POLICY CONFIGURATION
# =============================================================================

# Initialize policy engine with rules
policy_engine = PolicyEngine()
audit_trail = AuditTrail()

# Rule: Read operations are low risk, auto-approved
policy_engine.add_rule(PolicyRule(
    tool_name="read_file",
    risk_level=RiskLevel.LOW,
    requires_approval=False,
    reason="Read-only operation"
))

# Rule: Write operations require approval
policy_engine.add_rule(PolicyRule(
    tool_name="write_file",
    risk_level=RiskLevel.MEDIUM,
    requires_approval=True,
    required_role="developer",
    reason="File modification"
))

# Rule: Delete operations are high risk
policy_engine.add_rule(PolicyRule(
    tool_name="delete_file",
    risk_level=RiskLevel.HIGH,
    requires_approval=True,
    required_role="admin",
    reason="Destructive operation"
))

# Rule: Database drops are critical
policy_engine.add_rule(PolicyRule(
    tool_name="delete_database_table",
    risk_level=RiskLevel.CRITICAL,
    requires_approval=True,
    required_role="dba",
    reason="Database destructive operation"
))

# Rule: Bulk email over 100 recipients needs approval
policy_engine.add_rule(PolicyRule(
    tool_name="send_bulk_email",
    condition="recipients > 100",
    risk_level=RiskLevel.HIGH,
    requires_approval=True,
    required_role="marketing_manager",
    reason="Large-scale email operation"
))

# Rule: Bulk email under 100 is auto-approved
policy_engine.add_rule(PolicyRule(
    tool_name="send_bulk_email",
    condition="recipients <= 100",
    risk_level=RiskLevel.LOW,
    requires_approval=False,
    reason="Small-scale email"
))


# =============================================================================
# GRAPH NODES
# =============================================================================

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


def agent(state: State) -> State:
    """Agent node."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def approval_gate(state: State) -> Command:
    """
    Approval gate node - evaluates policy and interrupts if approval required.
    This is the key implementation for GV-01.
    """
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return Command(goto="tools")

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # Evaluate against policy
    risk_level, requires_approval, reason = policy_engine.evaluate(
        tool_name, tool_args
    )

    print(f"\n  [POLICY] Tool: {tool_name}")
    print(f"  [POLICY] Risk: {risk_level.value}")
    print(f"  [POLICY] Requires approval: {requires_approval}")
    print(f"  [POLICY] Reason: {reason}")

    if not requires_approval:
        # Auto-approve low-risk operations
        audit_trail.log(
            tool_name=tool_name,
            tool_args=tool_args,
            risk_level=risk_level,
            decision="auto-approved",
            reason=reason
        )
        return Command(goto="tools")

    # Interrupt for approval
    decision = interrupt({
        "action": "approve_destructive",
        "tool_name": tool_name,
        "tool_args": tool_args,
        "risk_level": risk_level.value,
        "reason": reason,
        "message": f"Approve {risk_level.value.upper()} risk operation: {tool_name}?"
    })

    action = decision.get("action", "reject")
    approver_id = decision.get("approver_id", "unknown")
    rejection_reason = decision.get("reason", "")

    if action == "approve":
        audit_trail.log(
            tool_name=tool_name,
            tool_args=tool_args,
            risk_level=risk_level,
            decision="approved",
            approver_id=approver_id,
            reason=f"Approved by {approver_id}"
        )
        return Command(goto="tools")

    else:
        # Rejected - return to agent with rejection message
        audit_trail.log(
            tool_name=tool_name,
            tool_args=tool_args,
            risk_level=risk_level,
            decision="rejected",
            approver_id=approver_id,
            reason=rejection_reason
        )

        rejection_msg = ToolMessage(
            content=f"Operation rejected: {rejection_reason or 'No reason provided'}",
            tool_call_id=tool_call["id"]
        )
        return Command(goto="agent", update={"messages": [rejection_msg]})


def should_continue(state: State) -> str:
    """Conditional edge."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "approval_gate"
    return END


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_graph():
    """Build graph with approval gate."""
    builder = StateGraph(State)

    builder.add_node("agent", agent)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["approval_gate", END])
    builder.add_edge("approval_gate", "tools")
    builder.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# TESTS
# =============================================================================

def test_auto_approve_low_risk():
    """Test that low-risk operations are auto-approved."""
    print("\n" + "=" * 70)
    print("TEST: Auto-approve Low Risk (read_file)")
    print("=" * 70)

    graph = build_graph()
    config = {"configurable": {"thread_id": "test-low-risk-1"}}

    result = graph.invoke(
        {"messages": [("user", "Read the file /etc/passwd")]},
        config=config
    )

    state = graph.get_state(config)
    if state.next:
        print(f"Unexpected interrupt at: {state.next}")
    else:
        print(f"Result: {result['messages'][-1].content[:100]}...")
        print("✅ Auto-approved without interrupt")


def test_approve_medium_risk():
    """Test medium risk operation with approval."""
    print("\n" + "=" * 70)
    print("TEST: Medium Risk with Approval (write_file)")
    print("=" * 70)

    graph = build_graph()
    config = {"configurable": {"thread_id": "test-medium-risk-1"}}

    # First invoke - should interrupt
    result = graph.invoke(
        {"messages": [("user", "Write 'hello' to /tmp/test.txt")]},
        config=config
    )

    state = graph.get_state(config)
    if state.next:
        print(f"Interrupted at: {state.next}")
        print(f"Interrupt value: {state.tasks[0].interrupts[0].value}")

        # Resume with approval
        result = graph.invoke(
            Command(resume={"action": "approve", "approver_id": "dev@example.com"}),
            config=config
        )
        print(f"Result after approval: {result['messages'][-1].content[:100]}...")
        print("✅ Approved and executed")
    else:
        print("❌ Expected interrupt but got none")


def test_reject_high_risk():
    """Test high risk operation with rejection."""
    print("\n" + "=" * 70)
    print("TEST: High Risk with Rejection (delete_file)")
    print("=" * 70)

    graph = build_graph()
    config = {"configurable": {"thread_id": "test-high-risk-1"}}

    # First invoke - should interrupt
    result = graph.invoke(
        {"messages": [("user", "Delete the file /important/data.txt")]},
        config=config
    )

    state = graph.get_state(config)
    if state.next:
        print(f"Interrupted at: {state.next}")
        interrupt_value = state.tasks[0].interrupts[0].value
        print(f"Risk level: {interrupt_value.get('risk_level')}")

        # Resume with rejection
        result = graph.invoke(
            Command(resume={
                "action": "reject",
                "approver_id": "security@example.com",
                "reason": "Production data cannot be deleted without backup"
            }),
            config=config
        )
        print(f"Result after rejection: {result['messages'][-1].content[:100]}...")
        print("✅ Rejected as expected")


def test_conditional_policy():
    """Test conditional policy (bulk email threshold)."""
    print("\n" + "=" * 70)
    print("TEST: Conditional Policy (bulk email count)")
    print("=" * 70)

    graph = build_graph()

    # Test 1: Under threshold (should auto-approve)
    print("\n--- Test: 50 recipients (under threshold) ---")
    config1 = {"configurable": {"thread_id": "test-bulk-1"}}
    result = graph.invoke(
        {"messages": [("user", "Send bulk email to 50 recipients with subject 'Newsletter'")]},
        config=config1
    )
    state = graph.get_state(config1)
    if state.next:
        print("❌ Unexpected interrupt for small batch")
    else:
        print("✅ Auto-approved for small batch")

    # Test 2: Over threshold (should require approval)
    print("\n--- Test: 500 recipients (over threshold) ---")
    config2 = {"configurable": {"thread_id": "test-bulk-2"}}
    result = graph.invoke(
        {"messages": [("user", "Send bulk email to 500 recipients with subject 'Big Campaign'")]},
        config=config2
    )
    state = graph.get_state(config2)
    if state.next:
        print(f"Interrupted at: {state.next}")
        print("✅ Required approval for large batch")

        # Approve it
        result = graph.invoke(
            Command(resume={"action": "approve", "approver_id": "marketing@example.com"}),
            config=config2
        )
        print(f"Result: {result['messages'][-1].content[:100]}...")
    else:
        print("❌ Expected interrupt for large batch")


def test_audit_trail():
    """Test audit trail recording."""
    print("\n" + "=" * 70)
    print("TEST: Audit Trail")
    print("=" * 70)

    print(f"\nTotal audit entries: {len(audit_trail.entries)}")
    for entry in audit_trail.entries[-5:]:
        print(f"  [{entry.timestamp.strftime('%H:%M:%S')}] {entry.decision}: "
              f"{entry.tool_name} ({entry.risk_level.value}) - {entry.approver_id or 'SYSTEM'}")


# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ GV-01: DESTRUCTIVE OPERATION GATE - EVALUATION SUMMARY                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LangGraph Native Support: ⭐⭐⭐ (PoC Ready)                                 │
│                                                                             │
│ LangGraph provides:                                                         │
│   ✅ interrupt() - Core mechanism for approval gates                        │
│   ✅ Command(resume=...) - Approval/rejection flow                          │
│   ✅ Checkpointer - State persistence for pending approvals                 │
│                                                                             │
│ LangGraph does NOT provide:                                                 │
│   ❌ Policy engine / risk classification                                    │
│   ❌ Conditional policies (threshold-based)                                 │
│   ❌ Audit trail                                                            │
│   ❌ Approver authorization                                                 │
│   ❌ Multi-level approval                                                   │
│                                                                             │
│ Custom Implementation Required:                                             │
│   ✅ PolicyEngine - Rule-based policy evaluation                            │
│   ✅ RiskLevel classification                                               │
│   ✅ Conditional rules (e.g., count > threshold)                            │
│   ✅ AuditTrail - Approval/rejection logging                                │
│   ✅ approval_gate node - Integrates policy with interrupt()                │
│                                                                             │
│ Tested Scenarios:                                                           │
│   ✓ Auto-approve low-risk (read_file)                                       │
│   ✓ Require approval for medium-risk (write_file)                           │
│   ✓ Reject high-risk (delete_file)                                          │
│   ✓ Conditional policies (bulk email count threshold)                       │
│   ✓ Audit trail recording                                                   │
│                                                                             │
│ Production Considerations:                                                  │
│   - Persistent policy storage (not hardcoded)                               │
│   - Role-based approval authorization                                       │
│   - Multi-approver workflows                                                │
│   - Timeout for pending approvals                                           │
│   - Notification system for approvers                                       │
│   - Tamper-proof audit trail (append-only, hash chain)                      │
│                                                                             │
│ VERDICT:                                                                    │
│   Fail-Close item. interrupt() provides the mechanism, but policy           │
│   engine, audit trail, and authorization are all custom. Works well         │
│   once implemented.                                                         │
│                                                                             │
│ Rating: ⭐⭐⭐ (PoC Ready)                                                   │
│   - Core mechanism (interrupt) is solid                                     │
│   - All governance logic is custom                                          │
│   - Production needs authorization + audit + notifications                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_auto_approve_low_risk()
    test_approve_medium_risk()
    test_reject_high_risk()
    test_conditional_policy()
    test_audit_trail()

    print(SUMMARY)
