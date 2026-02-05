"""
Human-in-the-Loop - Part 1: Approval Flow (HI-01, HI-03)
Native HITL API: needs_approval, RunState.approve/reject, state serialization

Updated for OpenAI Agents SDK v0.8.0 native HITL support.
"""

from dotenv import load_dotenv
load_dotenv()


import asyncio
import json
from agents import Agent, Runner, function_tool
from agents.run import RunState


# =============================================================================
# HI-01: Native Interrupt API via needs_approval
# =============================================================================

# Method 1: Always require approval (bool)
@function_tool(needs_approval=True)
def delete_file(path: str) -> str:
    """Delete a file. This operation always requires human approval."""
    return f"Deleted file: {path}"


@function_tool(needs_approval=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. This operation always requires human approval."""
    return f"Email sent to {to} with subject '{subject}'"


@function_tool(needs_approval=True)
def execute_query(sql: str) -> str:
    """Execute a database query. This operation always requires human approval."""
    return f"Query executed: {sql[:50]}..."


# Method 2: Conditional approval (callable)
async def needs_dangerous_approval(ctx, params: dict, call_id: str) -> bool:
    """Conditionally require approval for dangerous operations."""
    path = params.get("path", "")
    # Require approval for system directories or important files
    dangerous_paths = ["/etc", "/usr", "/important", "/production"]
    return any(danger in path for danger in dangerous_paths)


@function_tool(needs_approval=needs_dangerous_approval)
def read_file(path: str) -> str:
    """Read a file. Requires approval for sensitive paths."""
    return f"Contents of {path}: [file data here]"


# Safe tool (no approval needed)
@function_tool
def list_files(directory: str) -> str:
    """List files in a directory. No approval needed."""
    return f"Files in {directory}: file1.txt, file2.txt, file3.txt"


# =============================================================================
# Agent with HITL tools
# =============================================================================

hitl_agent = Agent(
    name="HITLBot",
    instructions="""You are an assistant that performs file and email operations.
    When asked to perform an operation, ALWAYS use the appropriate tool immediately.
    Do not ask for confirmation - the system handles approvals automatically.
    Just call the tool with the requested parameters.""",
    tools=[
        delete_file,
        send_email,
        execute_query,
        read_file,
        list_files,
    ],
)


# =============================================================================
# Tests
# =============================================================================

async def test_native_approval_flow():
    """Test native HITL approval flow (HI-01)."""
    print("\n" + "=" * 70)
    print("TEST: Native Approval Flow (HI-01)")
    print("=" * 70)

    # Run agent - will be interrupted due to needs_approval=True
    result = await Runner.run(
        hitl_agent,
        "Delete the file /tmp/test.txt"
    )

    # Check for interruptions directly on result
    has_interruptions = len(result.interruptions) > 0

    print(f"\nResult has interruptions: {has_interruptions}")
    print(f"Number of pending approvals: {len(result.interruptions)}")

    if has_interruptions:
        # Get state for approval handling
        state = result.to_state()

        for i, interruption in enumerate(result.interruptions):
            print(f"\n  Interruption {i + 1}:")
            print(f"    Tool: {interruption.raw_item.name}")
            print(f"    Call ID: {interruption.raw_item.call_id}")
            print(f"    Arguments: {interruption.raw_item.arguments}")

        # Approve all pending tool calls
        print("\n  [APPROVING all tool calls...]")
        for interruption in result.interruptions:
            state.approve(interruption)

        # Resume execution with approved state
        print("  [RESUMING execution...]")
        result = await Runner.run(hitl_agent, state)

        print(f"\nFinal output: {result.final_output}")
    else:
        print(f"\nNo approvals needed. Final output: {result.final_output}")

    print("\n✅ Native approval flow test completed")


async def test_rejection_flow():
    """Test rejection flow (HI-03)."""
    print("\n" + "=" * 70)
    print("TEST: Rejection Flow (HI-03)")
    print("=" * 70)

    result = await Runner.run(
        hitl_agent,
        "Execute this query: DROP TABLE users"
    )

    print(f"\nPending approvals: {len(result.interruptions)}")

    if result.interruptions:
        state = result.to_state()

        print("\n  [REJECTING dangerous operation...]")
        for interruption in result.interruptions:
            print(f"    Rejecting: {interruption.raw_item.name}")
            state.reject(interruption)

        # Resume with rejection
        result = await Runner.run(hitl_agent, state)
        print(f"\nFinal output after rejection: {result.final_output}")

    print("\n✅ Rejection flow test completed")


async def test_conditional_approval():
    """Test conditional approval with callable (HI-01)."""
    print("\n" + "=" * 70)
    print("TEST: Conditional Approval (HI-01)")
    print("=" * 70)

    # Safe path - should not require approval
    print("\n--- Safe path (no approval needed) ---")
    result = await Runner.run(
        hitl_agent,
        "Read the file /tmp/safe.txt"
    )
    print(f"Interruptions for safe path: {len(result.interruptions)}")
    if not result.interruptions:
        print(f"Output: {result.final_output}")

    # Dangerous path - should require approval
    print("\n--- Dangerous path (approval required) ---")
    result = await Runner.run(
        hitl_agent,
        "Read the file /etc/passwd"
    )
    print(f"Interruptions for dangerous path: {len(result.interruptions)}")

    if result.interruptions:
        state = result.to_state()
        for interruption in result.interruptions:
            print(f"  Approval needed for: {interruption.raw_item.name}")
            print(f"  Path: {interruption.raw_item.arguments}")
            state.approve(interruption)

        result = await Runner.run(hitl_agent, state)
        print(f"Output after approval: {result.final_output}")

    print("\n✅ Conditional approval test completed")


async def test_state_serialization():
    """Test state serialization for persistence (HI-01)."""
    print("\n" + "=" * 70)
    print("TEST: State Serialization (HI-01)")
    print("=" * 70)

    # Create interrupted state
    result = await Runner.run(
        hitl_agent,
        "Send email to admin@example.com with subject 'Alert' and body 'System alert'"
    )

    if result.interruptions:
        state = result.to_state()

        # Serialize state to JSON dict
        serialized = state.to_json()
        print(f"\nSerialized state type: {type(serialized)}")
        print(f"Serialized state keys: {list(serialized.keys()) if isinstance(serialized, dict) else 'N/A'}")

        # Convert to string for storage
        serialized_str = json.dumps(serialized)
        print(f"JSON string length: {len(serialized_str)} chars")
        print(f"JSON preview: {serialized_str[:200]}...")

        # In production: save to database, file, etc.
        # saved_state = db.save(serialized_str)

        # Later: restore state
        restored_dict = json.loads(serialized_str)
        restored_state = await RunState.from_json(hitl_agent, restored_dict)

        # Verify interruptions are preserved
        restored_interruptions = restored_state.get_interruptions()
        print(f"\nRestored interruptions: {len(restored_interruptions)}")

        # Approve and resume from restored state
        for interruption in restored_interruptions:
            restored_state.approve(interruption)

        result = await Runner.run(hitl_agent, restored_state)
        print(f"Final output after restore: {result.final_output}")
    else:
        print("\nNo interruptions - model may have responded without calling tool")

    print("\n✅ State serialization test completed")


async def test_multiple_approvals():
    """Test multiple tool calls requiring approval."""
    print("\n" + "=" * 70)
    print("TEST: Multiple Approvals")
    print("=" * 70)

    result = await Runner.run(
        hitl_agent,
        "Delete /tmp/file1.txt and send an email to user@example.com with subject 'Done' and body 'Files cleaned'"
    )

    print(f"\nTotal pending approvals: {len(result.interruptions)}")

    if result.interruptions:
        state = result.to_state()

        # Selectively approve/reject
        for i, interruption in enumerate(result.interruptions):
            tool_name = interruption.raw_item.name
            print(f"\n  [{i + 1}] {tool_name}")

            if tool_name == "delete_file":
                print("    -> Approving file deletion")
                state.approve(interruption)
            elif tool_name == "send_email":
                print("    -> Rejecting email (not needed)")
                state.reject(interruption)
            else:
                print("    -> Approving by default")
                state.approve(interruption)

        # Resume
        result = await Runner.run(hitl_agent, state)
        print(f"\nFinal output: {result.final_output}")

    print("\n✅ Selective approve/reject test completed")


def test_sync_wrapper():
    """Test synchronous wrapper for approval flow."""
    print("\n" + "=" * 70)
    print("TEST: Sync Wrapper (Runner.run_sync)")
    print("=" * 70)

    # Note: For sync usage, the approval loop needs to be handled differently
    # This is a simplified demo
    result = Runner.run_sync(
        hitl_agent,
        "List files in /tmp directory"  # This doesn't require approval
    )
    print(f"\nSync result: {result.final_output}")
    print(f"Interruptions in sync: {len(result.interruptions)}")
    print("\n✅ Sync wrapper test completed")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ HI-01, HI-03: NATIVE HITL API - EVALUATION SUMMARY                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ HI-01 (Interrupt API): ⭐⭐⭐⭐⭐ (Production Recommended)                    │
│   ✅ Native needs_approval=True parameter                                   │
│   ✅ Conditional approval via callable                                      │
│   ✅ result.interruptions for pending approvals                             │
│   ✅ State serialization with to_json()/from_json()                         │
│   ✅ Seamless resume with Runner.run(agent, state)                          │
│                                                                             │
│ HI-03 (Resume Control): ⭐⭐⭐⭐⭐ (Production Recommended)                   │
│   ✅ state.approve(interruption) for approval                               │
│   ✅ state.reject(interruption) for rejection                               │
│   ✅ Selective approve/reject per tool call                                 │
│   ✅ Resume execution preserves context                                     │
│                                                                             │
│ Key Features (v0.8.0+):                                                     │
│   - @function_tool(needs_approval=True) - always require approval           │
│   - @function_tool(needs_approval=callable) - conditional approval          │
│   - result.interruptions - list of pending approvals                        │
│   - result.to_state() - get RunState for manipulation                       │
│   - state.approve(item) / state.reject(item) - make decisions               │
│   - state.to_json() / RunState.from_json(agent, data) - persistence         │
│   - Runner.run(agent, state) - resume from state                            │
│                                                                             │
│ Comparison with LangGraph:                                                  │
│   LangGraph:                                                                │
│     - interrupt() pauses execution                                          │
│     - Command(resume=...) resumes with data                                 │
│     - Checkpointer stores interrupted state                                 │
│   OpenAI SDK (v0.8.0+):                                                     │
│     - needs_approval parameter (cleaner!)                                   │
│     - RunState.approve/reject (explicit!)                                   │
│     - JSON serialization (portable!)                                        │
│     >>> NOW COMPARABLE TO LANGGRAPH <<<                                     │
│                                                                             │
│ Production Usage:                                                           │
│   1. Mark sensitive tools with needs_approval=True                          │
│   2. Run agent, check result.interruptions                                  │
│   3. Serialize state.to_json(), store in DB                                 │
│   4. Present approval UI to human                                           │
│   5. Restore with RunState.from_json(agent, data)                           │
│   6. Apply decisions, resume with Runner.run(agent, state)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    asyncio.run(test_native_approval_flow())
    asyncio.run(test_rejection_flow())
    asyncio.run(test_conditional_approval())
    asyncio.run(test_state_serialization())
    asyncio.run(test_multiple_approvals())
    test_sync_wrapper()

    print(SUMMARY)
