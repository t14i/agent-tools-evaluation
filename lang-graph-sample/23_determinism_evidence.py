"""
LangGraph Determinism - Evidence Collection & Non-determinism Isolation (DR-02, DR-03)
Capturing evidence for decisions and isolating non-deterministic components.

Evaluation: DR-02 (Evidence Reference), DR-03 (Non-determinism Isolation)
"""

import hashlib
import json
from typing import Annotated, TypedDict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# =============================================================================
# EVIDENCE COLLECTION (DR-02)
# =============================================================================

class EvidenceType(Enum):
    """Types of evidence."""
    USER_INPUT = "user_input"
    LLM_RESPONSE = "llm_response"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"
    EXTERNAL_DATA = "external_data"
    DECISION = "decision"


@dataclass
class Evidence:
    """A piece of evidence for a decision."""
    id: str
    type: EvidenceType
    timestamp: datetime
    content: Any
    content_hash: str
    source: str  # Where this evidence came from
    metadata: dict = field(default_factory=dict)


@dataclass
class DecisionRecord:
    """Record of a decision with supporting evidence."""
    decision_id: str
    timestamp: datetime
    node_name: str
    decision_type: str  # "tool_selection", "routing", "approval"
    decision_value: Any
    evidence_ids: list[str]  # References to evidence
    rationale: Optional[str] = None


class EvidenceCollector:
    """
    Collects and manages evidence for decisions.
    Implements DR-02: Evidence Reference.
    """

    def __init__(self):
        self.evidence: dict[str, Evidence] = {}
        self.decisions: list[DecisionRecord] = []
        self.current_chain: list[str] = []  # Evidence chain for current execution

    def collect(
        self,
        type: EvidenceType,
        content: Any,
        source: str,
        metadata: dict = None
    ) -> Evidence:
        """Collect a piece of evidence."""
        content_str = json.dumps(content, sort_keys=True, default=str)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

        evidence_id = f"ev_{content_hash}_{datetime.now().strftime('%H%M%S%f')}"

        evidence = Evidence(
            id=evidence_id,
            type=type,
            timestamp=datetime.now(),
            content=content,
            content_hash=content_hash,
            source=source,
            metadata=metadata or {}
        )

        self.evidence[evidence_id] = evidence
        self.current_chain.append(evidence_id)

        print(f"  [EVIDENCE] Collected: {type.value} from {source}")

        return evidence

    def record_decision(
        self,
        node_name: str,
        decision_type: str,
        decision_value: Any,
        rationale: Optional[str] = None
    ) -> DecisionRecord:
        """Record a decision with its supporting evidence."""
        decision_id = f"dec_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        record = DecisionRecord(
            decision_id=decision_id,
            timestamp=datetime.now(),
            node_name=node_name,
            decision_type=decision_type,
            decision_value=decision_value,
            evidence_ids=self.current_chain.copy(),
            rationale=rationale
        )

        self.decisions.append(record)

        # Reset chain for next decision
        self.current_chain = []

        print(f"  [DECISION] Recorded: {decision_type} = {decision_value}")
        print(f"    Evidence chain: {len(record.evidence_ids)} items")

        return record

    def get_evidence_for_decision(self, decision_id: str) -> list[Evidence]:
        """Get all evidence for a specific decision."""
        for decision in self.decisions:
            if decision.decision_id == decision_id:
                return [self.evidence[eid] for eid in decision.evidence_ids if eid in self.evidence]
        return []

    def get_decision_trace(self) -> list[dict]:
        """Get full decision trace with evidence."""
        trace = []
        for decision in self.decisions:
            trace.append({
                "decision_id": decision.decision_id,
                "timestamp": decision.timestamp.isoformat(),
                "node": decision.node_name,
                "type": decision.decision_type,
                "value": decision.decision_value,
                "rationale": decision.rationale,
                "evidence": [
                    {
                        "id": self.evidence[eid].id,
                        "type": self.evidence[eid].type.value,
                        "source": self.evidence[eid].source,
                        "hash": self.evidence[eid].content_hash
                    }
                    for eid in decision.evidence_ids
                    if eid in self.evidence
                ]
            })
        return trace


# =============================================================================
# NON-DETERMINISM ISOLATION (DR-03)
# =============================================================================

class ExecutionMode(Enum):
    """Execution modes for non-determinism control."""
    NORMAL = "normal"  # Normal execution with LLM
    DETERMINISTIC = "deterministic"  # Use cached responses only
    RECORDING = "recording"  # Normal + record for later replay


class DeterministicController:
    """
    Controls non-deterministic components.
    Implements DR-03: Non-determinism Isolation.
    """

    def __init__(self):
        self.mode = ExecutionMode.NORMAL
        self.llm_cache: dict[str, str] = {}  # input_hash -> response
        self.tool_cache: dict[str, str] = {}  # input_hash -> response
        self.blocked_nodes: set[str] = set()  # Nodes blocked in deterministic mode

    def set_mode(self, mode: ExecutionMode):
        """Set execution mode."""
        self.mode = mode
        print(f"  [MODE] Execution mode set to: {mode.value}")

    def block_node(self, node_name: str):
        """Block a node from executing in deterministic mode."""
        self.blocked_nodes.add(node_name)

    def is_blocked(self, node_name: str) -> bool:
        """Check if a node is blocked."""
        if self.mode == ExecutionMode.DETERMINISTIC:
            return node_name in self.blocked_nodes
        return False

    def cache_llm_response(self, input_hash: str, response: str):
        """Cache an LLM response."""
        self.llm_cache[input_hash] = response

    def get_cached_llm_response(self, input_hash: str) -> Optional[str]:
        """Get cached LLM response if in deterministic mode."""
        if self.mode == ExecutionMode.DETERMINISTIC:
            return self.llm_cache.get(input_hash)
        return None

    def should_use_cache(self) -> bool:
        """Check if cache should be used."""
        return self.mode == ExecutionMode.DETERMINISTIC

    def should_record(self) -> bool:
        """Check if responses should be recorded."""
        return self.mode in [ExecutionMode.NORMAL, ExecutionMode.RECORDING]

    @staticmethod
    def hash_input(content: Any) -> str:
        """Create hash of input content."""
        serialized = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


# =============================================================================
# STATE AND TOOLS
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    execution_mode: str
    evidence_chain: list


# Initialize systems
evidence_collector = EvidenceCollector()
deterministic_controller = DeterministicController()


@tool
def search_database(query: str) -> str:
    """Search the database (deterministic - same query = same result)."""
    # Simulated database search
    results = {
        "users": "Found 42 users matching criteria",
        "orders": "Found 128 orders in the last month",
        "products": "Found 15 products in inventory"
    }
    for key, result in results.items():
        if key in query.lower():
            return result
    return f"Search results for '{query}': 10 records found"


@tool
def get_current_time() -> str:
    """Get current time (non-deterministic)."""
    return f"Current time: {datetime.now().isoformat()}"


@tool
def generate_random_id() -> str:
    """Generate a random ID (non-deterministic)."""
    import random
    return f"Generated ID: {random.randint(10000, 99999)}"


@tool
def fetch_external_api(endpoint: str) -> str:
    """Fetch from external API (non-deterministic)."""
    return f"API response from {endpoint}: {{'status': 'ok', 'data': [...]}}"


tools = [search_database, get_current_time, generate_random_id, fetch_external_api]

# Mark non-deterministic tools
NON_DETERMINISTIC_TOOLS = {"get_current_time", "generate_random_id", "fetch_external_api"}


# =============================================================================
# GRAPH NODES
# =============================================================================

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


def agent_with_evidence(state: State) -> State:
    """Agent node with evidence collection and determinism control."""
    messages = state["messages"]

    # Collect user input as evidence
    if messages and hasattr(messages[-1], "content"):
        evidence_collector.collect(
            type=EvidenceType.USER_INPUT,
            content=messages[-1].content,
            source="user"
        )

    # Check execution mode
    input_hash = deterministic_controller.hash_input([str(m) for m in messages])

    if deterministic_controller.should_use_cache():
        cached = deterministic_controller.get_cached_llm_response(input_hash)
        if cached:
            print("  [DETERMINISTIC] Using cached LLM response")
            return {"messages": [AIMessage(content=cached)]}
        else:
            print("  [DETERMINISTIC] No cached response, blocking LLM")
            return {"messages": [AIMessage(content="[BLOCKED: No cached response available]")]}

    # Normal execution
    response = llm_with_tools.invoke(messages)

    # Collect LLM response as evidence
    evidence_collector.collect(
        type=EvidenceType.LLM_RESPONSE,
        content=response.content,
        source="llm",
        metadata={"model": "claude-sonnet-4-20250514", "input_hash": input_hash}
    )

    # Record the decision if tools were selected
    if response.tool_calls:
        evidence_collector.record_decision(
            node_name="agent",
            decision_type="tool_selection",
            decision_value=[tc["name"] for tc in response.tool_calls],
            rationale="LLM selected tools based on user request"
        )

    # Cache response if recording
    if deterministic_controller.should_record():
        deterministic_controller.cache_llm_response(input_hash, response.content)

    return {"messages": [response]}


class EvidenceToolNode:
    """Tool node that collects evidence for tool inputs/outputs."""

    def __init__(self, tools: list, evidence_collector: EvidenceCollector, controller: DeterministicController):
        self.tool_node = ToolNode(tools)
        self.tools_by_name = {t.name: t for t in tools}
        self.evidence_collector = evidence_collector
        self.controller = controller

    def __call__(self, state: dict) -> dict:
        """Execute tools with evidence collection."""
        last_message = state["messages"][-1]
        results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # Collect tool input as evidence
            self.evidence_collector.collect(
                type=EvidenceType.TOOL_INPUT,
                content={"tool": tool_name, "args": tool_args},
                source=f"tool:{tool_name}"
            )

            # Check if tool is non-deterministic and mode is deterministic
            is_non_deterministic = tool_name in NON_DETERMINISTIC_TOOLS
            input_hash = self.controller.hash_input({"tool": tool_name, "args": tool_args})

            if is_non_deterministic and self.controller.should_use_cache():
                cached = self.controller.tool_cache.get(input_hash)
                if cached:
                    print(f"  [DETERMINISTIC] Using cached result for {tool_name}")
                    result = cached
                else:
                    print(f"  [DETERMINISTIC] Blocking non-deterministic tool: {tool_name}")
                    result = f"[BLOCKED: Non-deterministic tool '{tool_name}' not allowed in deterministic mode]"
            else:
                # Execute tool
                tool_fn = self.tools_by_name[tool_name]
                result = tool_fn.invoke(tool_args)

                # Cache if recording
                if is_non_deterministic and self.controller.should_record():
                    self.controller.tool_cache[input_hash] = result

            # Collect tool output as evidence
            self.evidence_collector.collect(
                type=EvidenceType.TOOL_OUTPUT,
                content=result,
                source=f"tool:{tool_name}",
                metadata={"non_deterministic": is_non_deterministic}
            )

            results.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        return {"messages": results}


def should_continue(state: State) -> str:
    """Conditional edge."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_graph():
    """Build graph with evidence collection and determinism control."""
    builder = StateGraph(State)

    builder.add_node("agent", agent_with_evidence)
    builder.add_node("tools", EvidenceToolNode(tools, evidence_collector, deterministic_controller))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# TESTS
# =============================================================================

def test_evidence_collection():
    """Test evidence collection (DR-02)."""
    print("\n" + "=" * 70)
    print("TEST: Evidence Collection (DR-02)")
    print("=" * 70)

    # Clear previous evidence
    evidence_collector.evidence.clear()
    evidence_collector.decisions.clear()
    evidence_collector.current_chain.clear()

    graph = build_graph()
    config = {"configurable": {"thread_id": "evidence-test-1"}}

    deterministic_controller.set_mode(ExecutionMode.RECORDING)

    result = graph.invoke(
        {
            "messages": [("user", "Search the database for users")],
            "execution_mode": "recording",
            "evidence_chain": []
        },
        config=config
    )

    print(f"\n--- Evidence Collected ---")
    for ev_id, ev in evidence_collector.evidence.items():
        print(f"  {ev.type.value}: {str(ev.content)[:50]}... (hash: {ev.content_hash})")

    print(f"\n--- Decisions Recorded ---")
    for dec in evidence_collector.decisions:
        print(f"  {dec.decision_type}: {dec.decision_value}")
        print(f"    Evidence: {len(dec.evidence_ids)} items")


def test_decision_trace():
    """Test decision trace retrieval."""
    print("\n" + "=" * 70)
    print("TEST: Decision Trace")
    print("=" * 70)

    trace = evidence_collector.get_decision_trace()
    print(f"\nDecision trace ({len(trace)} decisions):")
    for dec in trace:
        print(f"\n  Decision: {dec['type']} = {dec['value']}")
        print(f"  Node: {dec['node']}")
        print(f"  Evidence:")
        for ev in dec['evidence']:
            print(f"    - {ev['type']} from {ev['source']}")


def test_deterministic_mode():
    """Test deterministic mode (DR-03)."""
    print("\n" + "=" * 70)
    print("TEST: Deterministic Mode (DR-03)")
    print("=" * 70)

    graph = build_graph()

    # Phase 1: Record execution
    print("\n--- Phase 1: Recording Mode ---")
    deterministic_controller.set_mode(ExecutionMode.RECORDING)

    config1 = {"configurable": {"thread_id": "deterministic-test-1"}}
    result1 = graph.invoke(
        {
            "messages": [("user", "Get the current time")],
            "execution_mode": "recording",
            "evidence_chain": []
        },
        config=config1
    )
    print(f"Result: {result1['messages'][-1].content[:100]}...")

    # Phase 2: Replay in deterministic mode
    print("\n--- Phase 2: Deterministic Mode ---")
    deterministic_controller.set_mode(ExecutionMode.DETERMINISTIC)

    config2 = {"configurable": {"thread_id": "deterministic-test-2"}}
    result2 = graph.invoke(
        {
            "messages": [("user", "Get the current time")],
            "execution_mode": "deterministic",
            "evidence_chain": []
        },
        config=config2
    )
    print(f"Result: {result2['messages'][-1].content[:100]}...")

    # Reset
    deterministic_controller.set_mode(ExecutionMode.NORMAL)


def test_non_deterministic_isolation():
    """Test non-deterministic tool isolation."""
    print("\n" + "=" * 70)
    print("TEST: Non-Deterministic Tool Isolation (DR-03)")
    print("=" * 70)

    print("\nNon-deterministic tools identified:")
    for tool_name in NON_DETERMINISTIC_TOOLS:
        print(f"  - {tool_name}")

    print("\nDeterministic tools:")
    for t in tools:
        if t.name not in NON_DETERMINISTIC_TOOLS:
            print(f"  - {t.name}")


def test_evidence_for_decision():
    """Test retrieving evidence for a specific decision."""
    print("\n" + "=" * 70)
    print("TEST: Evidence Retrieval for Decision")
    print("=" * 70)

    if evidence_collector.decisions:
        decision = evidence_collector.decisions[0]
        evidence = evidence_collector.get_evidence_for_decision(decision.decision_id)

        print(f"\nDecision: {decision.decision_type} = {decision.decision_value}")
        print(f"Supporting evidence ({len(evidence)} items):")
        for ev in evidence:
            print(f"  [{ev.type.value}] {str(ev.content)[:60]}...")
    else:
        print("No decisions recorded yet. Run test_evidence_collection first.")


# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ DR-02 & DR-03: EVIDENCE & DETERMINISM - EVALUATION SUMMARY                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LangGraph Native Support: ⭐ (Not Supported)                                │
│                                                                             │
│ LangGraph does NOT provide:                                                 │
│   ❌ Evidence collection                                                    │
│   ❌ Decision tracing                                                       │
│   ❌ Non-determinism tagging                                                │
│   ❌ Deterministic execution mode                                           │
│   ❌ Response caching for replay                                            │
│                                                                             │
│ Custom Implementation Required:                                             │
│   ✅ Evidence - Structured evidence with hashing                            │
│   ✅ EvidenceCollector - Collect and link evidence to decisions             │
│   ✅ DecisionRecord - Link decisions to evidence chain                      │
│   ✅ DeterministicController - Mode switching and caching                   │
│   ✅ EvidenceToolNode - Tool wrapper with evidence collection               │
│                                                                             │
│ DR-02 (Evidence Reference) Features:                                        │
│   ✓ Collect evidence for each component (input, LLM, tool)                  │
│   ✓ Hash evidence for integrity                                             │
│   ✓ Link evidence to decisions                                              │
│   ✓ Query evidence by decision                                              │
│   ✓ Full decision trace                                                     │
│                                                                             │
│ DR-03 (Non-determinism Isolation) Features:                                 │
│   ✓ Identify non-deterministic tools                                        │
│   ✓ Execution modes (normal, deterministic, recording)                      │
│   ✓ Cache LLM and tool responses                                            │
│   ✓ Block non-deterministic components                                      │
│   ✓ Replay with cached responses                                            │
│                                                                             │
│ Production Considerations:                                                  │
│   - Persistent evidence store                                               │
│   - Evidence retention policies                                             │
│   - Integration with compliance systems                                     │
│   - Distributed caching (Redis)                                             │
│   - Evidence search/query API                                               │
│   - Export for audit purposes                                               │
│                                                                             │
│ Rating:                                                                     │
│   DR-02 (Evidence): ⭐ (fully custom)                                       │
│   DR-03 (Non-determinism Isolation): ⭐ (fully custom)                      │
│                                                                             │
│   Important for incident investigation but requires full custom build.      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_evidence_collection()
    test_decision_trace()
    test_deterministic_mode()
    test_non_deterministic_isolation()
    test_evidence_for_decision()

    print(SUMMARY)
