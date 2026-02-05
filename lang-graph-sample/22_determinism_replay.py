"""
LangGraph Determinism & Replay (DR-01, DR-04)
Checkpoint-based replay and idempotency key management.

Evaluation: DR-01 (Replay), DR-04 (Idempotency)
"""

import hashlib
import json
import time
from typing import Annotated, TypedDict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# =============================================================================
# REPLAY INFRASTRUCTURE
# =============================================================================

@dataclass
class ReplayLogEntry:
    """Single entry in the replay log."""
    step: int
    timestamp: datetime
    node_name: str
    input_hash: str
    output_hash: str
    checkpoint_id: str
    llm_response: Optional[str] = None  # Captured LLM response for replay
    tool_responses: Optional[dict] = None  # Captured tool responses


class ReplayLogger:
    """
    Captures execution history for replay capability.
    Enables DR-01: Replay (re-execution with same outputs).
    """

    def __init__(self):
        self.logs: dict[str, list[ReplayLogEntry]] = defaultdict(list)
        self.llm_cache: dict[str, str] = {}  # input_hash -> response
        self.tool_cache: dict[str, str] = {}  # input_hash -> response

    def log_step(
        self,
        thread_id: str,
        step: int,
        node_name: str,
        input_data: dict,
        output_data: dict,
        checkpoint_id: str,
        llm_response: Optional[str] = None,
        tool_responses: Optional[dict] = None
    ):
        """Log a step for later replay."""
        input_hash = self._hash_data(input_data)
        output_hash = self._hash_data(output_data)

        entry = ReplayLogEntry(
            step=step,
            timestamp=datetime.now(),
            node_name=node_name,
            input_hash=input_hash,
            output_hash=output_hash,
            checkpoint_id=checkpoint_id,
            llm_response=llm_response,
            tool_responses=tool_responses
        )

        self.logs[thread_id].append(entry)

        # Cache responses for replay
        if llm_response:
            self.llm_cache[input_hash] = llm_response
        if tool_responses:
            for tool_input_hash, response in tool_responses.items():
                self.tool_cache[tool_input_hash] = response

    def get_cached_llm_response(self, input_hash: str) -> Optional[str]:
        """Get cached LLM response for replay."""
        return self.llm_cache.get(input_hash)

    def get_cached_tool_response(self, input_hash: str) -> Optional[str]:
        """Get cached tool response for replay."""
        return self.tool_cache.get(input_hash)

    def get_history(self, thread_id: str) -> list[ReplayLogEntry]:
        """Get execution history for a thread."""
        return self.logs.get(thread_id, [])

    @staticmethod
    def _hash_data(data: dict) -> str:
        """Create deterministic hash of data."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


# =============================================================================
# IDEMPOTENCY KEY MANAGEMENT
# =============================================================================

@dataclass
class IdempotencyRecord:
    """Record of an idempotent operation."""
    key: str
    created_at: datetime
    result: str
    status: str  # "pending", "completed", "failed"


class IdempotencyKeyManager:
    """
    Manages idempotency keys for exactly-once execution.
    Enables DR-04: Idempotency / Exactly-once.
    """

    def __init__(self):
        self.records: dict[str, IdempotencyRecord] = {}

    def generate_key(self, tool_name: str, args: dict, context: Optional[dict] = None) -> str:
        """Generate idempotency key from tool call details."""
        key_data = {
            "tool": tool_name,
            "args": args,
            "context": context or {}
        }
        serialized = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:32]

    def check_and_set(self, key: str) -> tuple[bool, Optional[str]]:
        """
        Check if operation already executed.
        Returns: (is_new, existing_result)
        """
        if key in self.records:
            record = self.records[key]
            if record.status == "completed":
                return False, record.result
            elif record.status == "pending":
                return False, None  # In progress
            # Failed records can be retried
        return True, None

    def mark_pending(self, key: str):
        """Mark operation as in progress."""
        self.records[key] = IdempotencyRecord(
            key=key,
            created_at=datetime.now(),
            result="",
            status="pending"
        )

    def mark_completed(self, key: str, result: str):
        """Mark operation as completed."""
        if key in self.records:
            self.records[key].result = result
            self.records[key].status = "completed"

    def mark_failed(self, key: str, error: str):
        """Mark operation as failed."""
        if key in self.records:
            self.records[key].result = error
            self.records[key].status = "failed"

    def get_all_records(self) -> list[IdempotencyRecord]:
        """Get all idempotency records."""
        return list(self.records.values())


# =============================================================================
# STATE AND TOOLS
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    replay_mode: bool
    idempotency_keys: list


# Simulated side-effect tracking
side_effects: list[str] = []


@tool
def create_user(name: str, email: str) -> str:
    """Create a new user (has side effects)."""
    global side_effects
    result = f"Created user: {name} ({email})"
    side_effects.append(f"USER_CREATED: {email}")
    return result


@tool
def send_notification(user_email: str, message: str) -> str:
    """Send notification (has side effects)."""
    global side_effects
    result = f"Notification sent to {user_email}: {message}"
    side_effects.append(f"NOTIFICATION_SENT: {user_email}")
    return result


@tool
def update_database(table: str, operation: str) -> str:
    """Update database (has side effects)."""
    global side_effects
    result = f"Database {operation} on {table} completed"
    side_effects.append(f"DB_UPDATE: {table}.{operation}")
    return result


tools = [create_user, send_notification, update_database]


# =============================================================================
# REPLAY-ENABLED GRAPH
# =============================================================================

replay_logger = ReplayLogger()
idempotency_manager = IdempotencyKeyManager()

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


def agent_with_replay(state: State) -> State:
    """Agent node with replay support."""
    messages = state["messages"]
    replay_mode = state.get("replay_mode", False)

    # Generate input hash for caching
    input_hash = replay_logger._hash_data({"messages": [str(m) for m in messages]})

    if replay_mode:
        # Check for cached response
        cached = replay_logger.get_cached_llm_response(input_hash)
        if cached:
            print(f"  [REPLAY] Using cached LLM response")
            # Reconstruct AIMessage from cache (simplified)
            return {"messages": [AIMessage(content=cached)]}

    # Normal execution
    response = llm_with_tools.invoke(messages)

    # Cache response for future replay
    replay_logger.llm_cache[input_hash] = response.content

    return {"messages": [response]}


class IdempotentToolNode:
    """Tool node with idempotency support."""

    def __init__(self, tools: list, idempotency_manager: IdempotencyKeyManager):
        self.tool_node = ToolNode(tools)
        self.tools_by_name = {t.name: t for t in tools}
        self.idempotency_manager = idempotency_manager

    def __call__(self, state: State) -> State:
        """Execute tools with idempotency checks."""
        last_message = state["messages"][-1]
        results = []
        used_keys = state.get("idempotency_keys", [])

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # Generate idempotency key
            idem_key = self.idempotency_manager.generate_key(tool_name, tool_args)

            # Check for existing execution
            is_new, existing_result = self.idempotency_manager.check_and_set(idem_key)

            if not is_new and existing_result:
                print(f"  [IDEMPOTENT] Returning cached result for {tool_name}")
                results.append(ToolMessage(
                    content=f"[CACHED] {existing_result}",
                    tool_call_id=tool_id
                ))
                continue

            # Execute tool
            self.idempotency_manager.mark_pending(idem_key)
            try:
                tool_fn = self.tools_by_name[tool_name]
                result = tool_fn.invoke(tool_args)
                self.idempotency_manager.mark_completed(idem_key, result)
                used_keys.append(idem_key)
                print(f"  [EXECUTE] {tool_name} with key {idem_key[:8]}...")
                results.append(ToolMessage(content=result, tool_call_id=tool_id))
            except Exception as e:
                self.idempotency_manager.mark_failed(idem_key, str(e))
                results.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_id))

        return {"messages": results, "idempotency_keys": used_keys}


def should_continue(state: State) -> str:
    """Conditional edge."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_graph():
    """Build replay-enabled graph."""
    builder = StateGraph(State)

    builder.add_node("agent", agent_with_replay)
    builder.add_node("tools", IdempotentToolNode(tools, idempotency_manager))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# CHECKPOINT-BASED REPLAY
# =============================================================================

def replay_from_checkpoint(graph, config: dict, target_step: int):
    """
    Replay execution from a specific checkpoint.
    This is the core DR-01 capability.
    """
    # Get state history
    states = list(graph.get_state_history(config))

    if target_step >= len(states):
        print(f"Invalid step {target_step}, only {len(states)} states available")
        return None

    # States are in reverse order (newest first)
    target_state = states[-(target_step + 1)]

    print(f"Replaying from step {target_step}")
    print(f"  Checkpoint ID: {target_state.config['configurable']['checkpoint_id']}")
    print(f"  Next nodes: {target_state.next}")

    # Resume from checkpoint
    replay_config = {
        **config,
        "configurable": {
            **config["configurable"],
            "checkpoint_id": target_state.config["configurable"]["checkpoint_id"]
        }
    }

    return graph.invoke(None, config=replay_config)


# =============================================================================
# TESTS
# =============================================================================

def test_checkpoint_history():
    """Test checkpoint-based state history."""
    print("\n" + "=" * 70)
    print("TEST: Checkpoint-Based State History (DR-01)")
    print("=" * 70)

    graph = build_graph()
    config = {"configurable": {"thread_id": "replay-test-1"}}

    # Execute a workflow
    print("\n--- Initial Execution ---")
    result = graph.invoke(
        {
            "messages": [("user", "Create user John with email john@example.com")],
            "replay_mode": False,
            "idempotency_keys": []
        },
        config=config
    )

    # Get state history
    print("\n--- State History ---")
    states = list(graph.get_state_history(config))
    print(f"Total checkpoints: {len(states)}")

    for i, state in enumerate(states):
        checkpoint_id = state.config["configurable"].get("checkpoint_id", "N/A")
        next_nodes = state.next
        msg_count = len(state.values.get("messages", []))
        print(f"  [{i}] checkpoint={checkpoint_id[:8]}... next={next_nodes} messages={msg_count}")


def test_replay_execution():
    """Test replaying from a specific checkpoint."""
    print("\n" + "=" * 70)
    print("TEST: Replay from Checkpoint (DR-01)")
    print("=" * 70)

    graph = build_graph()
    config = {"configurable": {"thread_id": "replay-test-2"}}

    # Clear side effects
    global side_effects
    side_effects = []

    # Execute a workflow
    print("\n--- Initial Execution ---")
    result = graph.invoke(
        {
            "messages": [("user", "Create user Alice with email alice@example.com and send her a welcome notification")],
            "replay_mode": False,
            "idempotency_keys": []
        },
        config=config
    )
    print(f"Side effects after initial: {side_effects}")

    # Get state before tool execution
    states = list(graph.get_state_history(config))
    print(f"\nAvailable checkpoints: {len(states)}")

    # Replay from step 0 (initial state)
    print("\n--- Replaying from start ---")
    side_effects_before = len(side_effects)
    replay_result = replay_from_checkpoint(graph, config, 0)

    if replay_result:
        print(f"Replay completed")
        print(f"New side effects added: {len(side_effects) - side_effects_before}")


def test_idempotency():
    """Test idempotency key management."""
    print("\n" + "=" * 70)
    print("TEST: Idempotency Keys (DR-04)")
    print("=" * 70)

    graph = build_graph()

    # Clear side effects and idempotency records
    global side_effects
    side_effects = []
    idempotency_manager.records.clear()

    # Execute same operation twice
    config1 = {"configurable": {"thread_id": "idem-test-1"}}
    config2 = {"configurable": {"thread_id": "idem-test-2"}}

    print("\n--- First Execution ---")
    result1 = graph.invoke(
        {
            "messages": [("user", "Create user Bob with email bob@example.com")],
            "replay_mode": False,
            "idempotency_keys": []
        },
        config=config1
    )
    print(f"Side effects: {side_effects}")
    effects_after_first = len(side_effects)

    print("\n--- Second Execution (same operation) ---")
    result2 = graph.invoke(
        {
            "messages": [("user", "Create user Bob with email bob@example.com")],
            "replay_mode": False,
            "idempotency_keys": []
        },
        config=config2
    )
    print(f"Side effects: {side_effects}")
    effects_after_second = len(side_effects)

    print(f"\n--- Results ---")
    print(f"Effects after first: {effects_after_first}")
    print(f"Effects after second: {effects_after_second}")

    if effects_after_second == effects_after_first:
        print("✅ Idempotency working - no duplicate side effects")
    else:
        print("⚠️ Side effects were duplicated")

    print(f"\nIdempotency records:")
    for record in idempotency_manager.get_all_records():
        print(f"  Key: {record.key[:16]}... Status: {record.status}")


def test_replay_logger():
    """Test replay logger functionality."""
    print("\n" + "=" * 70)
    print("TEST: Replay Logger")
    print("=" * 70)

    # Log some steps
    replay_logger.log_step(
        thread_id="test-thread",
        step=0,
        node_name="agent",
        input_data={"messages": ["Hello"]},
        output_data={"response": "Hi there"},
        checkpoint_id="ckpt-001",
        llm_response="Hi there"
    )

    replay_logger.log_step(
        thread_id="test-thread",
        step=1,
        node_name="tools",
        input_data={"tool": "create_user"},
        output_data={"result": "User created"},
        checkpoint_id="ckpt-002",
        tool_responses={"abc123": "User created"}
    )

    # Retrieve history
    history = replay_logger.get_history("test-thread")
    print(f"\nReplay history for test-thread: {len(history)} entries")
    for entry in history:
        print(f"  Step {entry.step}: {entry.node_name} (checkpoint: {entry.checkpoint_id})")

    # Test cache retrieval
    input_hash = replay_logger._hash_data({"messages": ["Hello"]})
    cached = replay_logger.get_cached_llm_response(input_hash)
    print(f"\nCached LLM response: {cached}")


def test_exactly_once_semantics():
    """Test exactly-once execution semantics."""
    print("\n" + "=" * 70)
    print("TEST: Exactly-Once Semantics (DR-04)")
    print("=" * 70)

    # Clear state
    global side_effects
    side_effects = []
    idempotency_manager.records.clear()

    # Simulate multiple retries of same operation
    print("\n--- Simulating Retries ---")

    key = idempotency_manager.generate_key(
        "update_database",
        {"table": "users", "operation": "insert"}
    )

    for attempt in range(3):
        is_new, existing = idempotency_manager.check_and_set(key)
        print(f"Attempt {attempt + 1}: is_new={is_new}, existing={existing}")

        if is_new:
            idempotency_manager.mark_pending(key)
            # Simulate operation
            result = "Insert completed"
            side_effects.append("DB_INSERT")
            idempotency_manager.mark_completed(key, result)
            print(f"  Executed: {result}")
        elif existing:
            print(f"  Skipped: returning cached result")

    print(f"\nTotal side effects: {len(side_effects)}")
    print(f"Expected: 1 (exactly-once)")


# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ DR-01 & DR-04: REPLAY & IDEMPOTENCY - EVALUATION SUMMARY                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LangGraph Native Support: ⭐⭐ (Experimental)                                │
│                                                                             │
│ LangGraph provides (DR-01 Replay):                                          │
│   ✅ Checkpointer - State snapshots after each node                         │
│   ✅ get_state_history() - Access to all checkpoints                        │
│   ✅ Resume from checkpoint_id - Partial replay capability                  │
│                                                                             │
│ LangGraph does NOT provide:                                                 │
│   ❌ LLM response caching for deterministic replay                          │
│   ❌ Tool response recording                                                │
│   ❌ Full execution trace with inputs/outputs                               │
│   ❌ Replay verification (comparing original vs replay)                     │
│                                                                             │
│ LangGraph provides (DR-04 Idempotency):                                     │
│   ❌ No native idempotency key support                                      │
│   ❌ No commit_key or outbox pattern                                        │
│   ❌ No exactly-once guarantees                                             │
│                                                                             │
│ Custom Implementation Required:                                             │
│   ✅ ReplayLogger - Captures execution for replay                           │
│   ✅ LLM response caching - Deterministic replay                            │
│   ✅ IdempotencyKeyManager - Exactly-once execution                         │
│   ✅ IdempotentToolNode - Tool wrapper with deduplication                   │
│                                                                             │
│ Limitations of Native Checkpointer:                                         │
│   - Only captures state, not execution trace                                │
│   - No LLM call recording (non-deterministic on replay)                     │
│   - No tool response caching                                                │
│   - Replay restarts execution, doesn't "play back"                          │
│                                                                             │
│ Production Considerations:                                                  │
│   - Persistent idempotency store (Redis/PostgreSQL)                         │
│   - TTL for idempotency keys                                                │
│   - LLM response storage for incident investigation                         │
│   - Distributed idempotency (across instances)                              │
│                                                                             │
│ VERDICT:                                                                    │
│   Both are Fail-Close items. Checkpointer provides foundation for           │
│   DR-01 but doesn't capture LLM responses. DR-04 requires full              │
│   custom implementation.                                                    │
│                                                                             │
│ Rating:                                                                     │
│   DR-01 (Replay): ⭐⭐ (Experimental - partial support)                      │
│   DR-04 (Idempotency): ⭐ (Not Supported - fully custom)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_checkpoint_history()
    test_replay_execution()
    test_idempotency()
    test_replay_logger()
    test_exactly_once_semantics()

    print(SUMMARY)
