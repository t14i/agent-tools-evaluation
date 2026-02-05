"""
17_governance_audit.py - Governance Audit (GV-04, GV-06)

Purpose: Verify PII redaction and audit trail
- GV-04: PII redaction and log masking
- GV-06: Complete audit trail with tamper resistance
- Approver, diff, rationale, result, timestamp logging
- Hash chain for tamper detection

LangGraph Comparison:
- Both require custom implementation for audit logging
- CrewAI has no built-in audit trail mechanism
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from enum import Enum

from pydantic import BaseModel, Field


# =============================================================================
# PII Redaction (GV-04)
# =============================================================================

class PIIType(str, Enum):
    """Types of PII that can be redacted."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    API_KEY = "api_key"
    PASSWORD = "password"


class RedactionRule(BaseModel):
    """Configuration for a PII redaction rule."""

    pii_type: PIIType
    pattern: str  # Regex pattern
    replacement: str = "[REDACTED]"
    enabled: bool = True


class PIIRedactor:
    """
    PII redaction engine.

    Masks sensitive information in logs and outputs to comply with
    privacy requirements (GV-04).
    """

    def __init__(self):
        self.rules: list[RedactionRule] = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default redaction rules for common PII types."""
        self.rules = [
            # Email addresses
            RedactionRule(
                pii_type=PIIType.EMAIL,
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                replacement="[EMAIL_REDACTED]"
            ),
            # Phone numbers (various formats)
            RedactionRule(
                pii_type=PIIType.PHONE,
                pattern=r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
                replacement="[PHONE_REDACTED]"
            ),
            # SSN
            RedactionRule(
                pii_type=PIIType.SSN,
                pattern=r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
                replacement="[SSN_REDACTED]"
            ),
            # Credit card numbers
            RedactionRule(
                pii_type=PIIType.CREDIT_CARD,
                pattern=r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                replacement="[CARD_REDACTED]"
            ),
            # IP addresses
            RedactionRule(
                pii_type=PIIType.IP_ADDRESS,
                pattern=r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                replacement="[IP_REDACTED]"
            ),
            # API keys (common formats)
            RedactionRule(
                pii_type=PIIType.API_KEY,
                pattern=r'\b(?:sk-|pk-|api[_-]?key[=:]\s*)[A-Za-z0-9]{20,}\b',
                replacement="[API_KEY_REDACTED]"
            ),
            # Bearer tokens
            RedactionRule(
                pii_type=PIIType.API_KEY,
                pattern=r'Bearer\s+[A-Za-z0-9._-]+',
                replacement="Bearer [TOKEN_REDACTED]"
            ),
            # Passwords in common formats
            RedactionRule(
                pii_type=PIIType.PASSWORD,
                pattern=r'(?:password|passwd|pwd)[=:]\s*\S+',
                replacement="[PASSWORD_REDACTED]",
            ),
        ]

    def add_rule(self, rule: RedactionRule):
        """Add a custom redaction rule."""
        self.rules.append(rule)

    def redact(self, text: str) -> tuple[str, list[dict]]:
        """
        Redact PII from text.

        Returns:
            (redacted_text, list of redactions applied)
        """
        redactions = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            matches = list(re.finditer(rule.pattern, text, re.IGNORECASE))
            for match in matches:
                redactions.append({
                    "type": rule.pii_type.value,
                    "position": match.span(),
                    "original_length": len(match.group()),
                })

            text = re.sub(rule.pattern, rule.replacement, text, flags=re.IGNORECASE)

        return text, redactions

    def redact_dict(self, data: dict, sensitive_keys: Optional[list[str]] = None) -> dict:
        """
        Redact PII from dictionary values.

        Also redacts values of keys in sensitive_keys list.
        """
        sensitive_keys = sensitive_keys or [
            "password", "secret", "token", "api_key", "apikey",
            "auth", "credential", "private_key"
        ]

        result = {}
        for key, value in data.items():
            # Check if key is sensitive
            if any(s in key.lower() for s in sensitive_keys):
                result[key] = "[REDACTED]"
            elif isinstance(value, str):
                result[key], _ = self.redact(value)
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value, sensitive_keys)
            elif isinstance(value, list):
                result[key] = [
                    self.redact_dict(v, sensitive_keys) if isinstance(v, dict)
                    else (self.redact(v)[0] if isinstance(v, str) else v)
                    for v in value
                ]
            else:
                result[key] = value

        return result


# =============================================================================
# Audit Trail (GV-06)
# =============================================================================

class AuditEventType(str, Enum):
    """Types of audit events."""
    OPERATION_START = "operation_start"
    OPERATION_COMPLETE = "operation_complete"
    OPERATION_FAILED = "operation_failed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    DATA_ACCESS = "data_access"
    DATA_MODIFY = "data_modify"
    POLICY_VIOLATION = "policy_violation"
    ERROR = "error"


class AuditEntry(BaseModel):
    """A single audit log entry."""

    id: str
    timestamp: str
    event_type: AuditEventType
    actor: str  # Who performed the action
    action: str  # What action was performed
    resource: Optional[str] = None  # What resource was affected
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    diff: Optional[dict] = None  # Before/after for changes
    rationale: Optional[str] = None  # Why the action was taken
    result: str  # success, failure, pending
    duration_ms: Optional[int] = None
    metadata: dict = {}
    previous_hash: str = ""  # For chain integrity
    entry_hash: str = ""  # Hash of this entry


class AuditTrail:
    """
    Tamper-resistant audit trail.

    Features:
    - Complete logging of all operations
    - Hash chain for tamper detection
    - PII redaction before logging
    - Structured audit entries
    """

    def __init__(self, log_path: Optional[Path] = None, redactor: Optional[PIIRedactor] = None):
        self.log_path = log_path or Path("./db/audit_trail.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.redactor = redactor or PIIRedactor()
        self.previous_hash = self._get_last_hash()
        self.entry_count = 0

    def _get_last_hash(self) -> str:
        """Get the hash of the last entry in the log."""
        if not self.log_path.exists():
            return "GENESIS"

        last_line = ""
        with open(self.log_path, "r") as f:
            for line in f:
                if line.strip():
                    last_line = line

        if last_line:
            try:
                entry = json.loads(last_line)
                return entry.get("entry_hash", "GENESIS")
            except json.JSONDecodeError:
                pass

        return "GENESIS"

    def _compute_hash(self, entry: AuditEntry) -> str:
        """Compute hash for an entry including previous hash."""
        data = entry.model_dump()
        data.pop("entry_hash", None)  # Exclude current hash from computation
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def _generate_id(self) -> str:
        """Generate unique entry ID."""
        self.entry_count += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"audit_{timestamp}_{self.entry_count:04d}"

    def log(
        self,
        event_type: AuditEventType,
        actor: str,
        action: str,
        resource: Optional[str] = None,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
        diff: Optional[dict] = None,
        rationale: Optional[str] = None,
        result: str = "success",
        duration_ms: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> AuditEntry:
        """
        Log an audit event.

        All data is redacted before logging.
        """
        # Redact input/output
        input_summary = None
        if input_data:
            if isinstance(input_data, dict):
                redacted = self.redactor.redact_dict(input_data)
                input_summary = json.dumps(redacted, default=str)[:500]
            else:
                input_summary, _ = self.redactor.redact(str(input_data))
                input_summary = input_summary[:500]

        output_summary = None
        if output_data:
            if isinstance(output_data, dict):
                redacted = self.redactor.redact_dict(output_data)
                output_summary = json.dumps(redacted, default=str)[:500]
            else:
                output_summary, _ = self.redactor.redact(str(output_data))
                output_summary = output_summary[:500]

        # Redact diff
        redacted_diff = None
        if diff:
            redacted_diff = self.redactor.redact_dict(diff)

        # Create entry
        entry = AuditEntry(
            id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            actor=actor,
            action=action,
            resource=resource,
            input_summary=input_summary,
            output_summary=output_summary,
            diff=redacted_diff,
            rationale=rationale,
            result=result,
            duration_ms=duration_ms,
            metadata=metadata or {},
            previous_hash=self.previous_hash,
        )

        # Compute and set hash
        entry.entry_hash = self._compute_hash(entry)
        self.previous_hash = entry.entry_hash

        # Write to log
        with open(self.log_path, "a") as f:
            f.write(entry.model_dump_json() + "\n")

        return entry

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """
        Verify the integrity of the audit trail.

        Returns:
            (is_valid, list of issues)
        """
        issues = []

        if not self.log_path.exists():
            return True, []

        previous_hash = "GENESIS"

        with open(self.log_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    entry = AuditEntry(**data)
                except Exception as e:
                    issues.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue

                # Check previous hash
                if entry.previous_hash != previous_hash:
                    issues.append(
                        f"Line {line_num}: Chain broken - "
                        f"expected {previous_hash[:16]}..., got {entry.previous_hash[:16]}..."
                    )

                # Verify entry hash
                expected_hash = self._compute_hash(entry)
                if entry.entry_hash != expected_hash:
                    issues.append(
                        f"Line {line_num}: Hash mismatch - entry may be tampered"
                    )

                previous_hash = entry.entry_hash

        return len(issues) == 0, issues

    def get_entries(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        actor: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
    ) -> list[AuditEntry]:
        """Query audit entries with filters."""
        entries = []

        if not self.log_path.exists():
            return entries

        with open(self.log_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    entry = AuditEntry(**data)

                    # Apply filters
                    if start_time:
                        entry_time = datetime.fromisoformat(entry.timestamp)
                        if entry_time < start_time:
                            continue

                    if end_time:
                        entry_time = datetime.fromisoformat(entry.timestamp)
                        if entry_time > end_time:
                            continue

                    if actor and entry.actor != actor:
                        continue

                    if event_type and entry.event_type != event_type:
                        continue

                    entries.append(entry)

                except Exception:
                    continue

        return entries


# =============================================================================
# Demonstrations
# =============================================================================

def demo_pii_redaction():
    """Demonstrate PII redaction."""
    print("=" * 60)
    print("Demo 1: PII Redaction (GV-04)")
    print("=" * 60)

    redactor = PIIRedactor()

    # Test texts with various PII
    test_texts = [
        "Contact john.doe@example.com for more info",
        "Call me at 555-123-4567 or (555) 987-6543",
        "SSN: 123-45-6789",
        "Card number: 4111-1111-1111-1111",
        "Server IP: 192.168.1.100",
        "API key: sk-abc123def456ghi789jkl012mno345",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc123",
        "password=secret123",
    ]

    print("\nRedacting PII from text samples:\n")
    for text in test_texts:
        redacted, redactions = redactor.redact(text)
        print(f"Original: {text}")
        print(f"Redacted: {redacted}")
        if redactions:
            print(f"Types found: {[r['type'] for r in redactions]}")
        print()


def demo_dict_redaction():
    """Demonstrate dictionary redaction."""
    print("\n" + "=" * 60)
    print("Demo 2: Dictionary Redaction")
    print("=" * 60)

    redactor = PIIRedactor()

    data = {
        "user": {
            "name": "John Doe",
            "email": "john.doe@company.com",
            "phone": "555-123-4567",
        },
        "credentials": {
            "api_key": "sk-secret123456789",
            "password": "hunter2",
        },
        "metadata": {
            "ip_address": "192.168.1.100",
            "session_token": "abc123xyz",
        },
    }

    print("\nOriginal data:")
    print(json.dumps(data, indent=2))

    redacted = redactor.redact_dict(data)
    print("\nRedacted data:")
    print(json.dumps(redacted, indent=2))


def demo_audit_logging():
    """Demonstrate audit logging."""
    print("\n" + "=" * 60)
    print("Demo 3: Audit Logging (GV-06)")
    print("=" * 60)

    # Use a temp file for demo
    audit_path = Path("./db/demo_audit_trail.jsonl")
    if audit_path.exists():
        audit_path.unlink()

    trail = AuditTrail(log_path=audit_path)

    # Log some events
    print("\nLogging audit events...")

    trail.log(
        event_type=AuditEventType.OPERATION_START,
        actor="user_123",
        action="delete_records",
        resource="customer_table",
        input_data={"filter": {"email": "test@example.com"}, "limit": 100},
        rationale="Cleanup inactive users",
    )

    trail.log(
        event_type=AuditEventType.APPROVAL_REQUESTED,
        actor="user_123",
        action="bulk_delete",
        resource="customer_table",
        input_data={"count": 50},
    )

    trail.log(
        event_type=AuditEventType.APPROVAL_GRANTED,
        actor="admin_456",
        action="approve_bulk_delete",
        resource="customer_table",
        rationale="Verified records are inactive for >2 years",
        result="approved",
    )

    trail.log(
        event_type=AuditEventType.OPERATION_COMPLETE,
        actor="user_123",
        action="delete_records",
        resource="customer_table",
        output_data={"deleted_count": 50, "duration_ms": 1234},
        diff={"before": {"record_count": 1000}, "after": {"record_count": 950}},
        result="success",
        duration_ms=1234,
    )

    print("Logged 4 audit events")


def demo_audit_integrity():
    """Demonstrate audit trail integrity verification."""
    print("\n" + "=" * 60)
    print("Demo 4: Audit Trail Integrity Verification")
    print("=" * 60)

    audit_path = Path("./db/demo_audit_trail.jsonl")
    trail = AuditTrail(log_path=audit_path)

    print("\nVerifying audit trail integrity...")
    is_valid, issues = trail.verify_integrity()

    if is_valid:
        print("Audit trail integrity: VALID")
        print("Hash chain is intact, no tampering detected.")
    else:
        print("Audit trail integrity: COMPROMISED")
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")


def demo_audit_query():
    """Demonstrate audit trail querying."""
    print("\n" + "=" * 60)
    print("Demo 5: Audit Trail Querying")
    print("=" * 60)

    audit_path = Path("./db/demo_audit_trail.jsonl")
    trail = AuditTrail(log_path=audit_path)

    print("\nQuerying all entries:")
    entries = trail.get_entries()
    for entry in entries:
        print(f"\n  ID: {entry.id}")
        print(f"  Time: {entry.timestamp}")
        print(f"  Actor: {entry.actor}")
        print(f"  Action: {entry.action}")
        print(f"  Event: {entry.event_type.value}")
        print(f"  Result: {entry.result}")
        if entry.rationale:
            print(f"  Rationale: {entry.rationale}")

    print("\n\nQuerying approval events only:")
    approval_entries = trail.get_entries(event_type=AuditEventType.APPROVAL_GRANTED)
    for entry in approval_entries:
        print(f"  {entry.timestamp}: {entry.actor} approved {entry.action}")


def demo_tamper_detection():
    """Demonstrate tamper detection."""
    print("\n" + "=" * 60)
    print("Demo 6: Tamper Detection")
    print("=" * 60)

    # Create a fresh audit trail
    tamper_path = Path("./db/tamper_test_audit.jsonl")
    if tamper_path.exists():
        tamper_path.unlink()

    trail = AuditTrail(log_path=tamper_path)

    # Log some events
    trail.log(
        event_type=AuditEventType.DATA_ACCESS,
        actor="user_1",
        action="read_sensitive_data",
        resource="secret_table",
    )
    trail.log(
        event_type=AuditEventType.DATA_ACCESS,
        actor="user_2",
        action="read_sensitive_data",
        resource="secret_table",
    )

    print("Original audit trail:")
    is_valid, _ = trail.verify_integrity()
    print(f"  Integrity: {'VALID' if is_valid else 'INVALID'}")

    # Tamper with the file (modify an entry)
    print("\nSimulating tampering (modifying an entry)...")
    with open(tamper_path, "r") as f:
        lines = f.readlines()

    if lines:
        # Modify the first entry
        entry = json.loads(lines[0])
        entry["actor"] = "malicious_user"  # Tamper!
        lines[0] = json.dumps(entry) + "\n"

        with open(tamper_path, "w") as f:
            f.writelines(lines)

    print("\nAfter tampering:")
    trail2 = AuditTrail(log_path=tamper_path)
    is_valid, issues = trail2.verify_integrity()
    print(f"  Integrity: {'VALID' if is_valid else 'INVALID'}")
    if issues:
        print("  Issues detected:")
        for issue in issues:
            print(f"    - {issue}")


def main():
    print("=" * 60)
    print("Governance Audit Verification (GV-04, GV-06)")
    print("=" * 60)
    print("""
This script verifies PII redaction and audit trail capabilities.

Verification Items:
- GV-04: PII Redaction
  - Email, phone, SSN, credit card masking
  - API key and password redaction
  - Sensitive field detection in dictionaries

- GV-06: Complete Audit Trail
  - All operations logged with timestamp
  - Actor, action, resource, rationale tracking
  - Before/after diff for changes
  - Hash chain for tamper detection
  - Integrity verification

Key Concepts:
- PIIRedactor: Configurable PII masking
- AuditTrail: Tamper-resistant logging
- AuditEntry: Structured audit record
- Hash chain: Each entry includes hash of previous

LangGraph Comparison:
- Neither has built-in PII redaction
- Neither has built-in audit trail
- Custom implementation required for both
""")

    # Run all demos
    demo_pii_redaction()
    demo_dict_redaction()
    demo_audit_logging()
    demo_audit_integrity()
    demo_audit_query()
    demo_tamper_detection()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
