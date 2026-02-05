"""
LangGraph Governance - Audit Trail & PII Redaction (GV-04, GV-06)
PII detection/redaction and tamper-resistant audit logging.

Evaluation: GV-04 (PII / Redaction), GV-06 (Audit Trail Completeness)
"""

import hashlib
import json
import re
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
# PII DETECTION AND REDACTION (GV-04)
# =============================================================================

class PIIType(Enum):
    """Types of PII to detect."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"


@dataclass
class PIIMatch:
    """A detected PII match."""
    type: PIIType
    original: str
    redacted: str
    start: int
    end: int


class PIIRedactor:
    """
    Detects and redacts PII from text.
    Implements GV-04: PII / Redaction.
    """

    def __init__(self):
        self.patterns = {
            PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            PIIType.PHONE: r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            PIIType.SSN: r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            PIIType.CREDIT_CARD: r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            PIIType.IP_ADDRESS: r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        }
        self.redaction_format = {
            PIIType.EMAIL: "[EMAIL_REDACTED]",
            PIIType.PHONE: "[PHONE_REDACTED]",
            PIIType.SSN: "[SSN_REDACTED]",
            PIIType.CREDIT_CARD: "[CC_REDACTED]",
            PIIType.IP_ADDRESS: "[IP_REDACTED]",
            PIIType.NAME: "[NAME_REDACTED]",
            PIIType.ADDRESS: "[ADDRESS_REDACTED]",
        }

    def detect(self, text: str) -> list[PIIMatch]:
        """Detect all PII in text."""
        matches = []
        for pii_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                matches.append(PIIMatch(
                    type=pii_type,
                    original=match.group(),
                    redacted=self.redaction_format[pii_type],
                    start=match.start(),
                    end=match.end()
                ))
        return matches

    def redact(self, text: str) -> tuple[str, list[PIIMatch]]:
        """Redact all PII from text."""
        matches = self.detect(text)
        redacted_text = text

        # Sort by position (reverse) to preserve indices during replacement
        for match in sorted(matches, key=lambda m: m.start, reverse=True):
            redacted_text = (
                redacted_text[:match.start] +
                match.redacted +
                redacted_text[match.end:]
            )

        return redacted_text, matches

    def redact_dict(self, data: dict, sensitive_fields: set[str] = None) -> dict:
        """Redact PII from dictionary values."""
        sensitive_fields = sensitive_fields or {"password", "secret", "token", "api_key"}
        redacted = {}

        for key, value in data.items():
            if key.lower() in sensitive_fields:
                redacted[key] = "[SENSITIVE_FIELD_REDACTED]"
            elif isinstance(value, str):
                redacted[key], _ = self.redact(value)
            elif isinstance(value, dict):
                redacted[key] = self.redact_dict(value, sensitive_fields)
            else:
                redacted[key] = value

        return redacted


# =============================================================================
# TAMPER-RESISTANT AUDIT TRAIL (GV-06)
# =============================================================================

@dataclass
class AuditEntry:
    """Single audit log entry with hash chain."""
    sequence: int
    timestamp: datetime
    event_type: str
    actor_id: str
    action: str
    resource: str
    details: dict
    outcome: str  # "success", "failure", "pending"
    previous_hash: str
    entry_hash: str = field(default="")

    def __post_init__(self):
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of this entry."""
        data = {
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "actor_id": self.actor_id,
            "action": self.action,
            "resource": self.resource,
            "details": self.details,
            "outcome": self.outcome,
            "previous_hash": self.previous_hash
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()


class AuditTrail:
    """
    Tamper-resistant audit trail with hash chain.
    Implements GV-06: Audit Trail Completeness.
    """

    def __init__(self):
        self.entries: list[AuditEntry] = []
        self.genesis_hash = hashlib.sha256(b"genesis").hexdigest()

    def log(
        self,
        event_type: str,
        actor_id: str,
        action: str,
        resource: str,
        details: dict,
        outcome: str = "success"
    ) -> AuditEntry:
        """Add a new audit entry."""
        previous_hash = (
            self.entries[-1].entry_hash
            if self.entries
            else self.genesis_hash
        )

        entry = AuditEntry(
            sequence=len(self.entries),
            timestamp=datetime.now(),
            event_type=event_type,
            actor_id=actor_id,
            action=action,
            resource=resource,
            details=details,
            outcome=outcome,
            previous_hash=previous_hash
        )

        self.entries.append(entry)
        return entry

    def verify_integrity(self) -> tuple[bool, Optional[int]]:
        """
        Verify the integrity of the entire audit trail.
        Returns: (is_valid, first_invalid_sequence)
        """
        if not self.entries:
            return True, None

        # Check genesis
        if self.entries[0].previous_hash != self.genesis_hash:
            return False, 0

        # Check chain
        for i, entry in enumerate(self.entries):
            # Recompute hash
            expected_hash = entry._compute_hash()
            if entry.entry_hash != expected_hash:
                return False, i

            # Check chain link
            if i > 0:
                if entry.previous_hash != self.entries[i - 1].entry_hash:
                    return False, i

        return True, None

    def get_entries(
        self,
        actor_id: Optional[str] = None,
        event_type: Optional[str] = None,
        resource: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> list[AuditEntry]:
        """Query audit entries with filters."""
        results = self.entries

        if actor_id:
            results = [e for e in results if e.actor_id == actor_id]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if resource:
            results = [e for e in results if resource in e.resource]
        if since:
            results = [e for e in results if e.timestamp >= since]

        return results

    def export_json(self) -> str:
        """Export audit trail as JSON."""
        return json.dumps([
            {
                "sequence": e.sequence,
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type,
                "actor_id": e.actor_id,
                "action": e.action,
                "resource": e.resource,
                "details": e.details,
                "outcome": e.outcome,
                "hash": e.entry_hash
            }
            for e in self.entries
        ], indent=2)


# =============================================================================
# STATE AND TOOLS
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    actor_id: str
    pii_detected: list


# Initialize systems
pii_redactor = PIIRedactor()
audit_trail = AuditTrail()


@tool
def store_customer_data(customer_id: str, name: str, email: str, phone: str) -> str:
    """Store customer information (contains PII)."""
    return f"Stored data for customer {customer_id}: {name}, {email}, {phone}"


@tool
def process_payment(card_number: str, amount: float) -> str:
    """Process a payment (contains PII)."""
    return f"Processed payment of ${amount} on card ending in {card_number[-4:]}"


@tool
def send_notification(recipient_email: str, message: str) -> str:
    """Send a notification (contains PII)."""
    return f"Notification sent to {recipient_email}: {message}"


@tool
def generate_report(data_source: str) -> str:
    """Generate a report from data source."""
    return f"Report generated from {data_source}"


tools = [store_customer_data, process_payment, send_notification, generate_report]


# =============================================================================
# GRAPH NODES
# =============================================================================

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


def agent(state: State) -> State:
    """Agent node."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def audit_and_redact(state: State) -> Command:
    """
    Audit and redact node - logs operations and redacts PII.
    Implements GV-04 (PII Redaction) and GV-06 (Audit Trail).
    """
    last_message = state["messages"][-1]
    actor_id = state.get("actor_id", "unknown")
    pii_detected = state.get("pii_detected", [])

    if not last_message.tool_calls:
        return Command(goto="tools")

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    print(f"\n  [AUDIT] Tool: {tool_name}")
    print(f"  [AUDIT] Actor: {actor_id}")

    # Step 1: Detect and redact PII in arguments (GV-04)
    redacted_args = pii_redactor.redact_dict(tool_args)
    original_args_str = json.dumps(tool_args)
    _, pii_matches = pii_redactor.redact(original_args_str)

    if pii_matches:
        print(f"  [PII] Detected {len(pii_matches)} PII items:")
        for match in pii_matches:
            print(f"    - {match.type.value}: {match.original[:10]}...")
            pii_detected.append({
                "type": match.type.value,
                "tool": tool_name,
                "timestamp": datetime.now().isoformat()
            })

    # Step 2: Log to audit trail (GV-06)
    audit_entry = audit_trail.log(
        event_type="tool_execution",
        actor_id=actor_id,
        action=tool_name,
        resource=str(redacted_args),  # Log redacted version
        details={
            "original_pii_count": len(pii_matches),
            "tool_args_redacted": redacted_args
        },
        outcome="pending"
    )

    print(f"  [AUDIT] Entry #{audit_entry.sequence}, Hash: {audit_entry.entry_hash[:16]}...")

    return Command(goto="tools", update={"pii_detected": pii_detected})


class AuditedToolNode:
    """Tool node that logs execution results to audit trail."""

    def __init__(self, tools: list, audit_trail: AuditTrail, pii_redactor: PIIRedactor):
        self.tool_node = ToolNode(tools)
        self.tools_by_name = {t.name: t for t in tools}
        self.audit_trail = audit_trail
        self.pii_redactor = pii_redactor

    def __call__(self, state: dict) -> dict:
        """Execute tools and log results."""
        last_message = state["messages"][-1]
        actor_id = state.get("actor_id", "unknown")
        results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            try:
                # Execute tool
                tool_fn = self.tools_by_name[tool_name]
                result = tool_fn.invoke(tool_args)

                # Redact PII from result
                redacted_result, pii_matches = self.pii_redactor.redact(result)

                # Log success
                self.audit_trail.log(
                    event_type="tool_result",
                    actor_id=actor_id,
                    action=f"{tool_name}_completed",
                    resource=tool_name,
                    details={
                        "result_redacted": redacted_result,
                        "pii_redacted_count": len(pii_matches)
                    },
                    outcome="success"
                )

                results.append(ToolMessage(content=result, tool_call_id=tool_id))

            except Exception as e:
                # Log failure
                self.audit_trail.log(
                    event_type="tool_error",
                    actor_id=actor_id,
                    action=f"{tool_name}_failed",
                    resource=tool_name,
                    details={"error": str(e)},
                    outcome="failure"
                )
                results.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_id))

        return {"messages": results}


def should_continue(state: State) -> str:
    """Conditional edge."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "audit_and_redact"
    return END


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_graph():
    """Build graph with audit and redaction."""
    builder = StateGraph(State)

    builder.add_node("agent", agent)
    builder.add_node("audit_and_redact", audit_and_redact)
    builder.add_node("tools", AuditedToolNode(tools, audit_trail, pii_redactor))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["audit_and_redact", END])
    builder.add_edge("audit_and_redact", "tools")
    builder.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# TESTS
# =============================================================================

def test_pii_detection():
    """Test PII detection (GV-04)."""
    print("\n" + "=" * 70)
    print("TEST: PII Detection (GV-04)")
    print("=" * 70)

    test_texts = [
        "Contact john.doe@example.com for more info",
        "Call me at 555-123-4567 or (555) 987-6543",
        "SSN: 123-45-6789",
        "Card: 4111-1111-1111-1111",
        "Server IP: 192.168.1.100",
        "Email john@test.com and phone 555-1234, SSN 111-22-3333"
    ]

    for text in test_texts:
        matches = pii_redactor.detect(text)
        print(f"\nText: {text[:50]}...")
        for match in matches:
            print(f"  Found {match.type.value}: {match.original}")


def test_pii_redaction():
    """Test PII redaction (GV-04)."""
    print("\n" + "=" * 70)
    print("TEST: PII Redaction (GV-04)")
    print("=" * 70)

    text = """
    Customer: John Doe
    Email: john.doe@example.com
    Phone: (555) 123-4567
    SSN: 123-45-6789
    Payment Card: 4111-1111-1111-1111
    """

    redacted, matches = pii_redactor.redact(text)
    print(f"Original:\n{text}")
    print(f"\nRedacted:\n{redacted}")
    print(f"\nPII items redacted: {len(matches)}")


def test_dict_redaction():
    """Test dictionary redaction."""
    print("\n" + "=" * 70)
    print("TEST: Dictionary Redaction")
    print("=" * 70)

    data = {
        "customer_id": "C123",
        "email": "user@example.com",
        "phone": "555-123-4567",
        "password": "secret123",
        "api_key": "sk-1234567890",
        "nested": {
            "ssn": "123-45-6789"
        }
    }

    redacted = pii_redactor.redact_dict(data)
    print(f"Original: {json.dumps(data, indent=2)}")
    print(f"\nRedacted: {json.dumps(redacted, indent=2)}")


def test_audit_trail():
    """Test audit trail with hash chain (GV-06)."""
    print("\n" + "=" * 70)
    print("TEST: Audit Trail with Hash Chain (GV-06)")
    print("=" * 70)

    # Clear and create fresh audit trail
    test_audit = AuditTrail()

    # Add entries
    test_audit.log("login", "user1", "authenticate", "/auth", {"ip": "192.168.1.1"}, "success")
    test_audit.log("tool_execution", "user1", "store_customer_data", "/api/customers", {"customer_id": "C123"}, "success")
    test_audit.log("tool_execution", "user1", "process_payment", "/api/payments", {"amount": 100}, "success")

    print("\nAudit Trail:")
    for entry in test_audit.entries:
        print(f"  #{entry.sequence}: {entry.action} by {entry.actor_id}")
        print(f"    Hash: {entry.entry_hash[:32]}...")
        print(f"    Previous: {entry.previous_hash[:32]}...")

    # Verify integrity
    is_valid, invalid_at = test_audit.verify_integrity()
    print(f"\nIntegrity check: {'✅ Valid' if is_valid else f'❌ Invalid at {invalid_at}'}")


def test_tamper_detection():
    """Test tamper detection in audit trail."""
    print("\n" + "=" * 70)
    print("TEST: Tamper Detection")
    print("=" * 70)

    # Create audit trail
    test_audit = AuditTrail()
    test_audit.log("action1", "user1", "do_something", "/resource", {}, "success")
    test_audit.log("action2", "user1", "do_more", "/resource", {}, "success")
    test_audit.log("action3", "user1", "finish", "/resource", {}, "success")

    # Verify before tampering
    is_valid, _ = test_audit.verify_integrity()
    print(f"Before tampering: {'✅ Valid' if is_valid else '❌ Invalid'}")

    # Tamper with an entry
    test_audit.entries[1].action = "TAMPERED_ACTION"

    # Verify after tampering
    is_valid, invalid_at = test_audit.verify_integrity()
    print(f"After tampering: {'✅ Valid' if is_valid else f'❌ Invalid at sequence {invalid_at}'}")


def test_audit_query():
    """Test audit trail querying."""
    print("\n" + "=" * 70)
    print("TEST: Audit Trail Querying")
    print("=" * 70)

    # Clear and populate
    global audit_trail
    audit_trail = AuditTrail()

    audit_trail.log("login", "user1", "authenticate", "/auth", {}, "success")
    audit_trail.log("tool_execution", "user1", "store_customer", "/customers", {}, "success")
    audit_trail.log("tool_execution", "user2", "process_payment", "/payments", {}, "success")
    audit_trail.log("tool_execution", "user1", "send_notification", "/notifications", {}, "failure")

    # Query by actor
    user1_entries = audit_trail.get_entries(actor_id="user1")
    print(f"\nEntries by user1: {len(user1_entries)}")

    # Query by event type
    tool_entries = audit_trail.get_entries(event_type="tool_execution")
    print(f"Tool execution entries: {len(tool_entries)}")

    # Export JSON
    print(f"\nJSON Export (first 500 chars):")
    print(audit_trail.export_json()[:500])


def test_integrated_audit():
    """Test integrated audit in graph execution."""
    print("\n" + "=" * 70)
    print("TEST: Integrated Audit")
    print("=" * 70)

    # Clear audit trail
    global audit_trail
    audit_trail = AuditTrail()

    graph = build_graph()
    config = {"configurable": {"thread_id": "audit-test-1"}}

    result = graph.invoke(
        {
            "messages": [("user", "Store customer John Doe with email john@example.com and phone 555-123-4567")],
            "actor_id": "operator@company.com",
            "pii_detected": []
        },
        config=config
    )

    print(f"\nPII detected: {result.get('pii_detected', [])}")
    print(f"\nAudit entries created: {len(audit_trail.entries)}")

    for entry in audit_trail.entries:
        print(f"  [{entry.event_type}] {entry.action}: {entry.outcome}")


# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ GV-04 & GV-06: AUDIT & PII REDACTION - EVALUATION SUMMARY                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LangGraph Native Support: ⭐ (Not Supported)                                │
│                                                                             │
│ LangGraph does NOT provide:                                                 │
│   ❌ PII detection                                                          │
│   ❌ Automatic redaction                                                    │
│   ❌ Audit logging                                                          │
│   ❌ Tamper-resistant storage                                               │
│   ❌ Log masking                                                            │
│                                                                             │
│ Custom Implementation Required:                                             │
│   ✅ PIIRedactor - Regex-based PII detection and redaction                  │
│   ✅ PIIType enum - Categorize PII types                                    │
│   ✅ AuditEntry - Structured log entry with hash                            │
│   ✅ AuditTrail - Hash chain for tamper detection                           │
│   ✅ AuditedToolNode - Tool wrapper with logging                            │
│                                                                             │
│ GV-04 (PII / Redaction) Features:                                           │
│   ✓ Email, phone, SSN, credit card detection                                │
│   ✓ IP address detection                                                    │
│   ✓ Configurable redaction format                                           │
│   ✓ Dictionary recursive redaction                                          │
│   ✓ Sensitive field protection                                              │
│                                                                             │
│ GV-06 (Audit Trail) Features:                                               │
│   ✓ Hash chain for integrity                                                │
│   ✓ Tamper detection                                                        │
│   ✓ Actor, action, resource, outcome tracking                               │
│   ✓ Queryable log entries                                                   │
│   ✓ JSON export                                                             │
│                                                                             │
│ Production Considerations:                                                  │
│   - Use ML-based PII detection (Presidio, etc.)                             │
│   - Persistent audit storage (append-only database)                         │
│   - External audit log service (SIEM integration)                           │
│   - Log rotation and retention policies                                     │
│   - Compliance reporting (GDPR, HIPAA)                                      │
│   - Real-time alerting on sensitive operations                              │
│                                                                             │
│ Rating:                                                                     │
│   GV-04 (PII Redaction): ⭐ (fully custom)                                  │
│   GV-06 (Audit Trail): ⭐ (fully custom)                                    │
│                                                                             │
│   Critical for compliance but requires complete custom implementation.      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_pii_detection()
    test_pii_redaction()
    test_dict_redaction()
    test_audit_trail()
    test_tamper_detection()
    test_audit_query()
    test_integrated_audit()

    print(SUMMARY)
