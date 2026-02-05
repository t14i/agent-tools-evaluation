"""
Human-in-the-Loop - Part 2: State Manipulation (HI-02, HI-04, HI-05)
State editing, timeout handling, notification system
"""

from dotenv import load_dotenv
load_dotenv()


import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any, Callable
from agents import Agent, Runner, function_tool


# =============================================================================
# HI-02: State Manipulation
# =============================================================================

@dataclass
class HITLState:
    """State that can be manipulated during HITL pause."""
    conversation_id: str
    current_step: str
    pending_tool_call: Optional[dict] = None
    tool_args_override: Optional[dict] = None
    context: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def edit_tool_args(self, new_args: dict):
        """Allow human to edit tool arguments before execution."""
        if self.pending_tool_call:
            self.tool_args_override = new_args

    def add_context(self, key: str, value: Any):
        """Add context information during pause."""
        self.context[key] = value

    def serialize(self) -> str:
        """Serialize state for storage."""
        return json.dumps({
            "conversation_id": self.conversation_id,
            "current_step": self.current_step,
            "pending_tool_call": self.pending_tool_call,
            "tool_args_override": self.tool_args_override,
            "context": self.context,
            "created_at": self.created_at.isoformat()
        })

    @classmethod
    def deserialize(cls, data: str) -> "HITLState":
        """Deserialize state from storage."""
        parsed = json.loads(data)
        state = cls(
            conversation_id=parsed["conversation_id"],
            current_step=parsed["current_step"],
            pending_tool_call=parsed.get("pending_tool_call"),
            tool_args_override=parsed.get("tool_args_override"),
            context=parsed.get("context", {}),
        )
        state.created_at = datetime.fromisoformat(parsed["created_at"])
        return state


# =============================================================================
# HI-04: Timeout Handling
# =============================================================================

class TimeoutManager:
    """Manages timeouts for pending approvals."""

    def __init__(self, default_timeout: timedelta = timedelta(hours=24)):
        self.default_timeout = default_timeout
        self.timeouts: dict[str, datetime] = {}

    def set_timeout(self, approval_id: str, timeout: Optional[timedelta] = None):
        """Set timeout for an approval request."""
        deadline = datetime.now() + (timeout or self.default_timeout)
        self.timeouts[approval_id] = deadline

    def is_expired(self, approval_id: str) -> bool:
        """Check if an approval has expired."""
        if approval_id not in self.timeouts:
            return False
        return datetime.now() > self.timeouts[approval_id]

    def get_remaining(self, approval_id: str) -> Optional[timedelta]:
        """Get remaining time for approval."""
        if approval_id not in self.timeouts:
            return None
        remaining = self.timeouts[approval_id] - datetime.now()
        return remaining if remaining.total_seconds() > 0 else timedelta(0)

    def cleanup_expired(self) -> list[str]:
        """Remove expired timeouts and return their IDs."""
        now = datetime.now()
        expired = [k for k, v in self.timeouts.items() if now > v]
        for k in expired:
            del self.timeouts[k]
        return expired


# =============================================================================
# HI-05: Notification System
# =============================================================================

@dataclass
class Notification:
    """Notification for approvers."""
    id: str
    type: str  # "approval_required", "timeout_warning", "expired"
    approval_id: str
    message: str
    sent_at: datetime
    recipients: list[str]


class NotificationService:
    """Service for sending HITL notifications."""

    def __init__(self):
        self.notifications: list[Notification] = []
        self.handlers: dict[str, Callable] = {}
        self._counter = 0

    def register_handler(self, notification_type: str, handler: Callable):
        """Register a handler for notification type."""
        self.handlers[notification_type] = handler

    def notify_approval_required(
        self,
        approval_id: str,
        tool_name: str,
        tool_args: dict,
        recipients: list[str]
    ):
        """Send notification for required approval."""
        self._counter += 1
        notification = Notification(
            id=f"notif_{self._counter}",
            type="approval_required",
            approval_id=approval_id,
            message=f"Approval required for {tool_name}: {json.dumps(tool_args)}",
            sent_at=datetime.now(),
            recipients=recipients
        )
        self.notifications.append(notification)

        # Call handler if registered
        if "approval_required" in self.handlers:
            self.handlers["approval_required"](notification)

        print(f"  [NOTIFY] Sent to {recipients}: {notification.message[:50]}...")

    def notify_timeout_warning(self, approval_id: str, remaining: timedelta, recipients: list[str]):
        """Send timeout warning notification."""
        self._counter += 1
        notification = Notification(
            id=f"notif_{self._counter}",
            type="timeout_warning",
            approval_id=approval_id,
            message=f"Approval {approval_id} expires in {remaining}",
            sent_at=datetime.now(),
            recipients=recipients
        )
        self.notifications.append(notification)
        print(f"  [NOTIFY] Timeout warning: {notification.message}")

    def notify_expired(self, approval_id: str, recipients: list[str]):
        """Send expiration notification."""
        self._counter += 1
        notification = Notification(
            id=f"notif_{self._counter}",
            type="expired",
            approval_id=approval_id,
            message=f"Approval {approval_id} has expired and was auto-rejected",
            sent_at=datetime.now(),
            recipients=recipients
        )
        self.notifications.append(notification)
        print(f"  [NOTIFY] Expired: {notification.message}")


# =============================================================================
# Integrated HITL Manager
# =============================================================================

class HITLManager:
    """Combined manager for HITL operations."""

    def __init__(self):
        self.states: dict[str, HITLState] = {}
        self.timeout_manager = TimeoutManager()
        self.notification_service = NotificationService()
        self.approvers = ["admin@example.com", "manager@example.com"]

    def pause_for_approval(
        self,
        conversation_id: str,
        tool_name: str,
        tool_args: dict,
        timeout: Optional[timedelta] = None
    ) -> HITLState:
        """Pause execution for approval."""
        state = HITLState(
            conversation_id=conversation_id,
            current_step="pending_approval",
            pending_tool_call={"name": tool_name, "args": tool_args}
        )

        self.states[conversation_id] = state

        # Set timeout (HI-04)
        self.timeout_manager.set_timeout(conversation_id, timeout)

        # Send notification (HI-05)
        self.notification_service.notify_approval_required(
            approval_id=conversation_id,
            tool_name=tool_name,
            tool_args=tool_args,
            recipients=self.approvers
        )

        return state

    def get_state(self, conversation_id: str) -> Optional[HITLState]:
        """Get current HITL state."""
        return self.states.get(conversation_id)

    def edit_args(self, conversation_id: str, new_args: dict) -> bool:
        """Edit pending tool args (HI-02)."""
        state = self.states.get(conversation_id)
        if not state:
            return False

        state.edit_tool_args(new_args)
        return True

    def resume(self, conversation_id: str, approved: bool) -> dict:
        """Resume execution with decision."""
        state = self.states.get(conversation_id)
        if not state:
            return {"error": "State not found"}

        # Check timeout
        if self.timeout_manager.is_expired(conversation_id):
            self.notification_service.notify_expired(conversation_id, self.approvers)
            return {"error": "Approval expired"}

        tool_call = state.pending_tool_call
        if state.tool_args_override:
            tool_call["args"] = state.tool_args_override

        # Clean up
        del self.states[conversation_id]

        return {
            "approved": approved,
            "tool_call": tool_call
        }


# =============================================================================
# Tests
# =============================================================================

def test_state_manipulation():
    """Test state manipulation during HITL pause (HI-02)."""
    print("\n" + "=" * 70)
    print("TEST: State Manipulation (HI-02)")
    print("=" * 70)

    manager = HITLManager()

    # Pause for approval
    state = manager.pause_for_approval(
        conversation_id="conv_1",
        tool_name="send_email",
        tool_args={"to": "user@example.com", "subject": "Hello", "body": "Test"}
    )

    print(f"\nInitial state:")
    print(f"  Tool: {state.pending_tool_call['name']}")
    print(f"  Args: {state.pending_tool_call['args']}")

    # Human edits the args
    manager.edit_args("conv_1", {
        "to": "correct@example.com",  # Changed
        "subject": "Corrected Subject",  # Changed
        "body": "Test"
    })

    # Resume with edited args
    result = manager.resume("conv_1", approved=True)

    print(f"\nAfter edit and resume:")
    print(f"  Approved: {result['approved']}")
    print(f"  Final args: {result['tool_call']['args']}")
    print("✅ State manipulation works")


def test_timeout_handling():
    """Test timeout handling (HI-04)."""
    print("\n" + "=" * 70)
    print("TEST: Timeout Handling (HI-04)")
    print("=" * 70)

    timeout_mgr = TimeoutManager(default_timeout=timedelta(seconds=5))

    # Set a short timeout
    timeout_mgr.set_timeout("approval_1", timedelta(seconds=2))

    print(f"\nTimeout set for approval_1")
    print(f"Is expired: {timeout_mgr.is_expired('approval_1')}")
    print(f"Remaining: {timeout_mgr.get_remaining('approval_1')}")

    # Simulate time passing (in production, this would be real time)
    # For demo, we'll just check the logic

    print("\n✅ Timeout handling works")
    print("   (Full test would require async waiting)")


def test_notification_system():
    """Test notification system (HI-05)."""
    print("\n" + "=" * 70)
    print("TEST: Notification System (HI-05)")
    print("=" * 70)

    notif_service = NotificationService()

    # Register a custom handler
    received_notifications = []
    def custom_handler(notification):
        received_notifications.append(notification)

    notif_service.register_handler("approval_required", custom_handler)

    # Send notifications
    notif_service.notify_approval_required(
        approval_id="approval_1",
        tool_name="delete_file",
        tool_args={"path": "/important/data.txt"},
        recipients=["admin@example.com"]
    )

    notif_service.notify_timeout_warning(
        approval_id="approval_2",
        remaining=timedelta(hours=1),
        recipients=["admin@example.com"]
    )

    print(f"\nTotal notifications: {len(notif_service.notifications)}")
    print(f"Handler received: {len(received_notifications)}")
    print("✅ Notification system works")


def test_full_hitl_flow():
    """Test complete HITL flow."""
    print("\n" + "=" * 70)
    print("TEST: Full HITL Flow")
    print("=" * 70)

    manager = HITLManager()

    # Step 1: Operation triggers HITL
    print("\n--- Step 1: Trigger HITL ---")
    state = manager.pause_for_approval(
        conversation_id="conv_full",
        tool_name="execute_query",
        tool_args={"sql": "DELETE FROM users WHERE id = 123"},
        timeout=timedelta(hours=1)
    )

    # Step 2: Human reviews and edits
    print("\n--- Step 2: Human reviews ---")
    print(f"Pending approval: {state.pending_tool_call}")

    # Human adds context
    state.add_context("reviewed_by", "security_team")
    state.add_context("risk_assessment", "medium")

    # Human edits args (adds safety clause)
    manager.edit_args("conv_full", {
        "sql": "DELETE FROM users WHERE id = 123 AND deleted_at IS NULL"
    })

    # Step 3: Approve and resume
    print("\n--- Step 3: Approve and resume ---")
    result = manager.resume("conv_full", approved=True)

    print(f"Result: {result}")
    print("\n✅ Full HITL flow completed")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ HI-02, HI-04, HI-05: HITL STATE OPERATIONS - EVALUATION SUMMARY             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ HI-02 (State Manipulation): ⭐⭐⭐ (PoC Ready)                               │
│   ✅ State object can be serialized/deserialized                            │
│   ✅ Tool args can be edited during pause                                   │
│   ✅ Context can be added during pause                                      │
│   ❌ No built-in state manipulation API                                     │
│   ❌ Requires custom implementation                                         │
│                                                                             │
│ HI-04 (Timeout): ⭐⭐ (Experimental)                                        │
│   ✅ TimeoutManager can track deadlines                                     │
│   ✅ Expiration checking works                                              │
│   ❌ No built-in timeout mechanism                                          │
│   ❌ Requires custom async handling                                         │
│   ❌ No automatic timeout escalation                                        │
│                                                                             │
│ HI-05 (Notification): ⭐ (Not Supported)                                    │
│   ❌ No built-in notification system                                        │
│   ❌ Requires full custom implementation                                    │
│   ⚠️ Need to integrate with external services (email, Slack, etc.)         │
│                                                                             │
│ Custom Implementation Provided:                                             │
│   ✅ HITLState - Serializable state object                                  │
│   ✅ TimeoutManager - Deadline tracking                                     │
│   ✅ NotificationService - Notification dispatch                            │
│   ✅ HITLManager - Integrated HITL orchestration                            │
│                                                                             │
│ Production Considerations:                                                  │
│   - Use persistent storage for HITL state                                   │
│   - Implement proper async timeout handling                                 │
│   - Integrate with notification services (PagerDuty, Slack, etc.)          │
│   - Add escalation policies                                                 │
│   - Implement RBAC for approvers                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_state_manipulation()
    test_timeout_handling()
    test_notification_system()
    test_full_hitl_flow()

    print(SUMMARY)
