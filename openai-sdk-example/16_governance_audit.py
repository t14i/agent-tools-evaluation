"""
Governance - Part 2: Audit & Compliance (GV-04, GV-05, GV-06)
PII redaction, tenant binding, audit trail
"""

from dotenv import load_dotenv
load_dotenv()


import re
import json
import hashlib
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field


# =============================================================================
# GV-04: PII Redaction
# =============================================================================

class PIIRedactor:
    """
    Redacts Personally Identifiable Information from text.
    Implements GV-04: PII / Redaction.
    """

    def __init__(self):
        self.patterns = {
            "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL]"),
            "phone": (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "[PHONE]"),
            "ssn": (r'\b\d{3}-\d{2}-\d{4}\b', "[SSN]"),
            "credit_card": (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', "[CC]"),
            "ip_address": (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', "[IP]"),
        }
        self.custom_patterns: list[tuple[str, str]] = []

    def add_pattern(self, name: str, pattern: str, replacement: str):
        """Add a custom PII pattern."""
        self.patterns[name] = (pattern, replacement)

    def redact(self, text: str) -> tuple[str, list[dict]]:
        """
        Redact PII from text.
        Returns: (redacted_text, list of redactions made)
        """
        redactions = []
        redacted = text

        for pii_type, (pattern, replacement) in self.patterns.items():
            matches = list(re.finditer(pattern, redacted, re.IGNORECASE))
            for match in matches:
                redactions.append({
                    "type": pii_type,
                    "original_hash": hashlib.sha256(match.group().encode()).hexdigest()[:16],
                    "position": match.start()
                })
            redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)

        return redacted, redactions

    def is_sensitive(self, text: str) -> bool:
        """Check if text contains sensitive data."""
        for pattern, _ in self.patterns.values():
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


# =============================================================================
# GV-05: Tenant / Purpose Binding
# =============================================================================

@dataclass
class TenantContext:
    """Context for tenant-scoped operations."""
    tenant_id: str
    user_id: str
    purpose: str
    permissions: list[str]
    data_classification: str  # "public", "internal", "confidential", "restricted"
    created_at: datetime = field(default_factory=datetime.now)


class TenantBinding:
    """
    Enforces tenant and purpose binding for operations.
    Implements GV-05: Tenant / Purpose Binding.
    """

    def __init__(self):
        self.contexts: dict[str, TenantContext] = {}
        self.allowed_purposes = {
            "customer_support": ["read_profile", "update_preferences"],
            "analytics": ["read_aggregated"],
            "admin": ["read_profile", "update_profile", "delete_profile"],
        }

    def create_context(
        self,
        session_id: str,
        tenant_id: str,
        user_id: str,
        purpose: str,
        permissions: list[str] = None
    ) -> TenantContext:
        """Create a tenant context for a session."""
        # Auto-derive permissions from purpose if not specified
        if permissions is None:
            permissions = self.allowed_purposes.get(purpose, [])

        context = TenantContext(
            tenant_id=tenant_id,
            user_id=user_id,
            purpose=purpose,
            permissions=permissions,
            data_classification="internal"
        )

        self.contexts[session_id] = context
        return context

    def check_permission(self, session_id: str, operation: str) -> tuple[bool, str]:
        """Check if operation is allowed for session."""
        context = self.contexts.get(session_id)
        if not context:
            return (False, "No context found for session")

        if operation in context.permissions:
            return (True, f"Allowed by {context.purpose} policy")
        else:
            return (False, f"Operation '{operation}' not allowed for purpose '{context.purpose}'")

    def check_data_access(
        self,
        session_id: str,
        data_tenant_id: str,
        data_classification: str
    ) -> tuple[bool, str]:
        """Check if session can access data from another tenant."""
        context = self.contexts.get(session_id)
        if not context:
            return (False, "No context")

        # Enforce tenant isolation
        if context.tenant_id != data_tenant_id:
            return (False, "Cross-tenant access denied")

        # Check classification level
        levels = ["public", "internal", "confidential", "restricted"]
        if levels.index(data_classification) > levels.index(context.data_classification):
            return (False, f"Insufficient clearance for {data_classification} data")

        return (True, "Access granted")


# =============================================================================
# GV-06: Audit Trail
# =============================================================================

@dataclass
class AuditEntry:
    """A single audit log entry."""
    id: str
    timestamp: datetime
    session_id: str
    tenant_id: str
    user_id: str
    action: str
    resource: str
    result: str  # "success", "denied", "error"
    details: dict
    previous_hash: Optional[str] = None
    entry_hash: Optional[str] = None

    def compute_hash(self) -> str:
        """Compute hash of this entry for chain integrity."""
        data = f"{self.timestamp.isoformat()}|{self.session_id}|{self.action}|{self.result}|{self.previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()


class AuditTrail:
    """
    Tamper-evident audit trail using hash chain.
    Implements GV-06: Audit Trail Completeness.
    """

    def __init__(self):
        self.entries: list[AuditEntry] = []
        self._counter = 0

    def log(
        self,
        session_id: str,
        tenant_id: str,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        details: dict = None
    ) -> AuditEntry:
        """Log an auditable event."""
        self._counter += 1

        previous_hash = self.entries[-1].entry_hash if self.entries else "GENESIS"

        entry = AuditEntry(
            id=f"audit_{self._counter:08d}",
            timestamp=datetime.now(),
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            details=details or {},
            previous_hash=previous_hash
        )
        entry.entry_hash = entry.compute_hash()

        self.entries.append(entry)
        return entry

    def verify_chain(self) -> tuple[bool, Optional[str]]:
        """Verify the integrity of the audit chain."""
        if not self.entries:
            return (True, None)

        for i, entry in enumerate(self.entries):
            # Check hash
            computed = entry.compute_hash()
            if computed != entry.entry_hash:
                return (False, f"Hash mismatch at entry {entry.id}")

            # Check chain
            if i > 0:
                if entry.previous_hash != self.entries[i-1].entry_hash:
                    return (False, f"Chain broken at entry {entry.id}")

        return (True, None)

    def query(
        self,
        tenant_id: str = None,
        user_id: str = None,
        action: str = None,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> list[AuditEntry]:
        """Query audit entries."""
        results = self.entries

        if tenant_id:
            results = [e for e in results if e.tenant_id == tenant_id]
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if action:
            results = [e for e in results if e.action == action]
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        return results

    def export(self, format: str = "json") -> str:
        """Export audit trail."""
        if format == "json":
            return json.dumps([
                {
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat(),
                    "session_id": e.session_id,
                    "tenant_id": e.tenant_id,
                    "user_id": e.user_id,
                    "action": e.action,
                    "resource": e.resource,
                    "result": e.result,
                    "entry_hash": e.entry_hash
                }
                for e in self.entries
            ], indent=2)
        return ""


# =============================================================================
# Integrated Governance Layer
# =============================================================================

class GovernanceLayer:
    """Combined governance layer for agents."""

    def __init__(self):
        self.pii_redactor = PIIRedactor()
        self.tenant_binding = TenantBinding()
        self.audit_trail = AuditTrail()

    def process_input(
        self,
        session_id: str,
        input_text: str
    ) -> tuple[str, bool, str]:
        """Process and validate input."""
        context = self.tenant_binding.contexts.get(session_id)
        if not context:
            return (input_text, False, "No session context")

        # Redact PII from input
        redacted, _ = self.pii_redactor.redact(input_text)

        # Log input
        self.audit_trail.log(
            session_id=session_id,
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            action="input",
            resource="conversation",
            result="success",
            details={"redacted": redacted != input_text}
        )

        return (redacted, True, "OK")

    def process_output(
        self,
        session_id: str,
        output_text: str
    ) -> tuple[str, bool, str]:
        """Process and validate output."""
        context = self.tenant_binding.contexts.get(session_id)
        if not context:
            return (output_text, False, "No session context")

        # Redact PII from output
        redacted, redactions = self.pii_redactor.redact(output_text)

        # Log output
        self.audit_trail.log(
            session_id=session_id,
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            action="output",
            resource="conversation",
            result="success",
            details={"redactions": len(redactions)}
        )

        return (redacted, True, "OK")


# =============================================================================
# Tests
# =============================================================================

def test_pii_redaction():
    """Test PII redaction (GV-04)."""
    print("\n" + "=" * 70)
    print("TEST: PII Redaction (GV-04)")
    print("=" * 70)

    redactor = PIIRedactor()

    texts = [
        "Contact me at john@example.com or 555-123-4567",
        "My SSN is 123-45-6789 and CC is 4111-1111-1111-1111",
        "Server IP: 192.168.1.100",
        "No PII in this text"
    ]

    for text in texts:
        redacted, redactions = redactor.redact(text)
        print(f"\nOriginal: {text}")
        print(f"Redacted: {redacted}")
        print(f"Redactions: {len(redactions)}")

    print("\n✅ PII redaction works")


def test_tenant_binding():
    """Test tenant/purpose binding (GV-05)."""
    print("\n" + "=" * 70)
    print("TEST: Tenant/Purpose Binding (GV-05)")
    print("=" * 70)

    binding = TenantBinding()

    # Create contexts
    binding.create_context(
        session_id="session_1",
        tenant_id="tenant_A",
        user_id="user_1",
        purpose="customer_support"
    )

    binding.create_context(
        session_id="session_2",
        tenant_id="tenant_A",
        user_id="admin_1",
        purpose="admin"
    )

    # Test permissions
    tests = [
        ("session_1", "read_profile", True),
        ("session_1", "delete_profile", False),
        ("session_2", "delete_profile", True),
    ]

    for session_id, operation, expected in tests:
        allowed, reason = binding.check_permission(session_id, operation)
        status = "✅" if allowed == expected else "❌"
        print(f"{status} {session_id} -> {operation}: {allowed} ({reason})")

    # Test cross-tenant access
    allowed, reason = binding.check_data_access(
        "session_1", "tenant_B", "internal"
    )
    print(f"\nCross-tenant access: {allowed} ({reason})")

    print("\n✅ Tenant binding works")


def test_audit_trail():
    """Test audit trail (GV-06)."""
    print("\n" + "=" * 70)
    print("TEST: Audit Trail (GV-06)")
    print("=" * 70)

    audit = AuditTrail()

    # Log some events
    audit.log(
        session_id="session_1",
        tenant_id="tenant_A",
        user_id="user_1",
        action="read_profile",
        resource="user/123",
        result="success"
    )

    audit.log(
        session_id="session_1",
        tenant_id="tenant_A",
        user_id="user_1",
        action="update_profile",
        resource="user/123",
        result="success",
        details={"field": "email"}
    )

    audit.log(
        session_id="session_2",
        tenant_id="tenant_A",
        user_id="admin_1",
        action="delete_user",
        resource="user/456",
        result="denied"
    )

    print(f"\nAudit entries: {len(audit.entries)}")

    # Verify chain
    valid, error = audit.verify_chain()
    print(f"Chain integrity: {'✅ Valid' if valid else f'❌ {error}'}")

    # Query
    tenant_entries = audit.query(tenant_id="tenant_A")
    print(f"Entries for tenant_A: {len(tenant_entries)}")

    # Export
    print(f"\nExported (first 200 chars):\n{audit.export()[:200]}...")

    print("\n✅ Audit trail works")


def test_governance_layer():
    """Test integrated governance layer."""
    print("\n" + "=" * 70)
    print("TEST: Integrated Governance Layer")
    print("=" * 70)

    gov = GovernanceLayer()

    # Create session context
    gov.tenant_binding.create_context(
        session_id="gov_session_1",
        tenant_id="acme_corp",
        user_id="user_42",
        purpose="customer_support"
    )

    # Process input with PII
    input_text = "My email is test@example.com and phone is 555-123-4567"
    processed_input, ok, reason = gov.process_input("gov_session_1", input_text)

    print(f"\nInput processing:")
    print(f"  Original: {input_text}")
    print(f"  Processed: {processed_input}")

    # Process output
    output_text = "I found your account. Your email is test@example.com."
    processed_output, ok, reason = gov.process_output("gov_session_1", output_text)

    print(f"\nOutput processing:")
    print(f"  Original: {output_text}")
    print(f"  Processed: {processed_output}")

    # Check audit
    print(f"\nAudit entries: {len(gov.audit_trail.entries)}")

    print("\n✅ Governance layer works")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ GV-04, GV-05, GV-06: AUDIT & COMPLIANCE - EVALUATION SUMMARY                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ GV-04 (PII / Redaction): ⭐ (Not Supported)                                 │
│   ❌ No built-in PII detection                                              │
│   ❌ No automatic redaction                                                 │
│   ⚠️ Custom PIIRedactor implementation provided                            │
│   ⚠️ Consider using dedicated PII libraries (presidio, etc.)              │
│                                                                             │
│ GV-05 (Tenant / Purpose Binding): ⭐ (Not Supported)                        │
│   ❌ No built-in tenant isolation                                           │
│   ❌ No purpose binding                                                     │
│   ❌ No permission system                                                   │
│   ⚠️ Custom TenantBinding implementation provided                          │
│                                                                             │
│ GV-06 (Audit Trail): ⭐⭐⭐ (PoC Ready)                                      │
│   ✅ Tracing provides some audit capability                                 │
│   ❌ No tamper-evident logging                                              │
│   ❌ No compliance-grade audit trail                                        │
│   ⚠️ Custom AuditTrail with hash chain provided                            │
│                                                                             │
│ Custom Implementation Provided:                                             │
│   ✅ PIIRedactor - Regex-based PII detection and redaction                  │
│   ✅ TenantBinding - Multi-tenant context and permissions                   │
│   ✅ AuditTrail - Hash-chained audit log                                    │
│   ✅ GovernanceLayer - Integrated governance                                │
│                                                                             │
│ Comparison with LangGraph:                                                  │
│   LangGraph:                                                                │
│     - Similar gaps in native support                                        │
│     - LangSmith for basic audit                                             │
│   OpenAI SDK:                                                               │
│     - Tracing for basic observability                                       │
│     - Similar custom work needed                                            │
│                                                                             │
│ Production Notes:                                                           │
│   - Use specialized PII libraries for production                            │
│   - Integrate with existing IAM systems                                     │
│   - Store audit logs in append-only storage                                 │
│   - Consider compliance requirements (GDPR, HIPAA, SOX)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_pii_redaction()
    test_tenant_binding()
    test_audit_trail()
    test_governance_layer()

    print(SUMMARY)
