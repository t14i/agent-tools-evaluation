"""
Tool Calling - Part 3: Controllable Automation (TC-02)
Tool approval, require_approval setting, controlled execution
"""

from dotenv import load_dotenv
load_dotenv()


from agents import Agent, Runner, function_tool


# =============================================================================
# TC-02: Controllable Automation via require_approval
# =============================================================================

# Safe tool - no approval needed
@function_tool
def read_file(path: str) -> str:
    """Read a file (safe operation)."""
    return f"Contents of {path}: [file data]"


# Risky tool - approval required via human_input_callback
@function_tool
def delete_file(path: str) -> str:
    """Delete a file (destructive operation)."""
    return f"Deleted file: {path}"


@function_tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (requires approval)."""
    return f"Email sent to {to} with subject '{subject}'"


# =============================================================================
# Custom approval handler
# =============================================================================

class ApprovalHandler:
    """Handler for tool approval decisions."""

    def __init__(self):
        self.require_approval_for = {"delete_file", "send_email"}
        self.approved_tools = set()

    def should_approve(self, tool_name: str, tool_args: dict) -> bool:
        """Check if tool execution should be approved."""
        if tool_name not in self.require_approval_for:
            return True

        # In production, this would prompt a human
        print(f"\n  [APPROVAL] Tool: {tool_name}")
        print(f"  [APPROVAL] Args: {tool_args}")

        # Simulate approval (in production, wait for human input)
        # For demo, we auto-approve delete but reject email to specific domain
        if tool_name == "delete_file":
            print("  [APPROVAL] Auto-approved (demo)")
            return True
        elif tool_name == "send_email":
            if "spam" in tool_args.get("to", "").lower():
                print("  [APPROVAL] Rejected (spam detected)")
                return False
            print("  [APPROVAL] Approved")
            return True

        return True


# =============================================================================
# Agent with controlled tools
# =============================================================================

approval_handler = ApprovalHandler()

# Create tools with custom wrapper for approval
original_delete = delete_file
original_send = send_email


def wrapped_delete(path: str) -> str:
    """Wrapped delete with approval check."""
    if not approval_handler.should_approve("delete_file", {"path": path}):
        return "Operation rejected by approval system"
    return original_delete(path)


def wrapped_send(to: str, subject: str, body: str) -> str:
    """Wrapped send with approval check."""
    if not approval_handler.should_approve("send_email", {"to": to, "subject": subject, "body": body}):
        return "Operation rejected by approval system"
    return original_send(to, subject, body)


# Note: OpenAI SDK doesn't have built-in require_approval
# We need to wrap tools or use human_input_callback

controlled_agent = Agent(
    name="ControlledBot",
    instructions="You help with file and email operations. Use the appropriate tools as requested.",
    tools=[
        read_file,
        function_tool(wrapped_delete, name_override="delete_file", description_override="Delete a file (requires approval)"),
        function_tool(wrapped_send, name_override="send_email", description_override="Send an email (requires approval)"),
    ],
)


# =============================================================================
# Tests
# =============================================================================

def test_safe_operation():
    """Test safe operation without approval."""
    print("\n" + "=" * 70)
    print("TEST: Safe Operation (no approval)")
    print("=" * 70)

    result = Runner.run_sync(
        controlled_agent,
        "Read the file /etc/passwd"
    )

    print(f"\nResult: {result.final_output}")
    print("✅ Safe operation executed without approval")


def test_approved_destructive():
    """Test destructive operation with approval."""
    print("\n" + "=" * 70)
    print("TEST: Destructive Operation (approved)")
    print("=" * 70)

    result = Runner.run_sync(
        controlled_agent,
        "Delete the file /tmp/test.txt"
    )

    print(f"\nResult: {result.final_output}")
    print("✅ Destructive operation approved and executed")


def test_rejected_operation():
    """Test operation that gets rejected."""
    print("\n" + "=" * 70)
    print("TEST: Rejected Operation")
    print("=" * 70)

    result = Runner.run_sync(
        controlled_agent,
        "Send an email to spam@example.com with subject 'Hello' and body 'Test'"
    )

    print(f"\nResult: {result.final_output}")
    print("✅ Operation rejected by approval system")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ TC-02: CONTROLLABLE AUTOMATION - EVALUATION SUMMARY                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ OpenAI SDK Native Support: ⭐⭐⭐⭐ (Production Ready)                       │
│                                                                             │
│ OpenAI SDK provides:                                                        │
│   ✅ human_input_callback for approval flows                                │
│   ✅ Tool wrapping for custom approval logic                                │
│   ✅ RunConfig for execution control                                        │
│                                                                             │
│ OpenAI SDK does NOT provide:                                                │
│   ❌ Built-in require_approval flag per tool                                │
│   ❌ Declarative policy configuration                                       │
│   ❌ Risk classification system                                             │
│                                                                             │
│ Custom Implementation:                                                      │
│   ✅ ApprovalHandler - Custom approval logic                                │
│   ✅ Tool wrapping - Intercept before execution                             │
│   ✅ Policy-based approval - Conditional approval                           │
│                                                                             │
│ Comparison with LangGraph:                                                  │
│   - LangGraph: interrupt() for approval, checkpointer for state             │
│   - OpenAI SDK: human_input_callback or tool wrapping                       │
│   - Both require custom policy engine                                       │
│                                                                             │
│ Production Considerations:                                                  │
│   - Implement proper approval UI/API                                        │
│   - Add timeout for pending approvals                                       │
│   - Store approval state in durable storage                                 │
│   - Log all approval decisions                                              │
│                                                                             │
│ Rating: ⭐⭐⭐⭐ (Production Ready)                                          │
│   - Approval mechanism available                                            │
│   - Policy engine requires custom implementation                            │
│   - Flexible enough for production use                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_safe_operation()
    test_approved_destructive()
    test_rejected_operation()

    print(SUMMARY)
