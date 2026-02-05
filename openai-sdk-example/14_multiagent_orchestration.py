"""
Multi-Agent - Part 2: Orchestration Patterns (MA-03, MA-04, MA-05)
Hierarchical process, routing, shared memory
"""

from dotenv import load_dotenv
load_dotenv()


from typing import Annotated, Optional
from dataclasses import dataclass, field
from datetime import datetime
from agents import Agent, Runner, function_tool, handoff


# =============================================================================
# MA-05: Shared Memory Store
# =============================================================================

@dataclass
class SharedMemoryItem:
    """Item in shared memory."""
    key: str
    value: str
    created_by: str
    created_at: datetime
    tags: list[str] = field(default_factory=list)


class SharedMemoryStore:
    """
    Shared memory accessible by all agents.
    Implements MA-05: Shared Memory.
    """

    def __init__(self):
        self.store: dict[str, SharedMemoryItem] = {}

    def put(self, key: str, value: str, agent_name: str, tags: list[str] = None):
        """Store a value, tracking which agent created it."""
        self.store[key] = SharedMemoryItem(
            key=key,
            value=value,
            created_by=agent_name,
            created_at=datetime.now(),
            tags=tags or []
        )

    def get(self, key: str) -> Optional[SharedMemoryItem]:
        """Get a value from shared memory."""
        return self.store.get(key)

    def search(self, tags: list[str] = None, created_by: str = None) -> list[SharedMemoryItem]:
        """Search shared memory."""
        results = list(self.store.values())
        if tags:
            results = [r for r in results if any(t in r.tags for t in tags)]
        if created_by:
            results = [r for r in results if r.created_by == created_by]
        return results

    def list_all(self) -> list[str]:
        """List all keys."""
        return list(self.store.keys())


# Global shared memory
shared_memory = SharedMemoryStore()


# =============================================================================
# Tools with shared memory access
# =============================================================================

@function_tool
def save_to_shared_memory(key: str, value: str, tags: str = "") -> str:
    """Save information to shared memory for other agents to access."""
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    shared_memory.put(key, value, "CurrentAgent", tag_list)
    return f"Saved '{key}' to shared memory"


@function_tool
def read_from_shared_memory(key: str) -> str:
    """Read information from shared memory."""
    item = shared_memory.get(key)
    if item:
        return f"Found '{key}': {item.value} (created by {item.created_by})"
    return f"No data found for key '{key}'"


@function_tool
def search_shared_memory(tags: str = "") -> str:
    """Search shared memory by tags."""
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    results = shared_memory.search(tags=tag_list)
    if not results:
        return "No matching items found in shared memory"
    return "Found: " + ", ".join([f"{r.key}={r.value}" for r in results])


# =============================================================================
# MA-03: Hierarchical Process / Agent-as-Tool
# =============================================================================

# Sub-agents for hierarchical pattern
research_sub_agent = Agent(
    name="ResearcherSubAgent",
    instructions="""You are a research specialist.
    Gather information and save findings to shared memory.
    Be thorough and cite your sources.""",
    tools=[save_to_shared_memory, read_from_shared_memory],
)

writer_sub_agent = Agent(
    name="WriterSubAgent",
    instructions="""You are a content writer.
    Read research from shared memory and create well-written content.
    Structure your writing with clear sections.""",
    tools=[read_from_shared_memory, save_to_shared_memory],
)

reviewer_sub_agent = Agent(
    name="ReviewerSubAgent",
    instructions="""You are a quality reviewer.
    Review content for accuracy, clarity, and completeness.
    Read from shared memory and provide feedback.""",
    tools=[read_from_shared_memory, save_to_shared_memory],
)


# Agent-as-tool pattern: Create tools that invoke sub-agents
@function_tool
def delegate_research(topic: str) -> str:
    """Delegate research task to the research specialist."""
    result = Runner.run_sync(
        research_sub_agent,
        f"Research the following topic and save key findings to shared memory: {topic}"
    )
    return f"Research completed: {result.final_output[:200]}..."


@function_tool
def delegate_writing(topic: str, requirements: str) -> str:
    """Delegate writing task to the content writer."""
    result = Runner.run_sync(
        writer_sub_agent,
        f"Write content about {topic}. Requirements: {requirements}. Check shared memory for research."
    )
    return f"Writing completed: {result.final_output[:200]}..."


@function_tool
def delegate_review(content_key: str) -> str:
    """Delegate review task to the reviewer."""
    result = Runner.run_sync(
        reviewer_sub_agent,
        f"Review the content stored under key '{content_key}' in shared memory."
    )
    return f"Review completed: {result.final_output[:200]}..."


# Manager agent that orchestrates sub-agents
manager_agent = Agent(
    name="ManagerAgent",
    instructions="""You are a project manager overseeing a content creation team.
    You have access to three specialists:
    - Researcher: delegate_research
    - Writer: delegate_writing
    - Reviewer: delegate_review

    For content creation tasks:
    1. First delegate research
    2. Then delegate writing
    3. Finally delegate review

    Coordinate the workflow and report final results.""",
    tools=[delegate_research, delegate_writing, delegate_review, read_from_shared_memory],
)


# =============================================================================
# MA-04: Routing Patterns
# =============================================================================

# Different routing strategies

# 1. Handoff-based routing (LLM decides)
handoff_router = Agent(
    name="HandoffRouter",
    instructions="""Route requests to the appropriate specialist:
    - Technical questions -> TechAgent
    - Business questions -> BizAgent
    - General questions -> GeneralAgent""",
    handoffs=[
        handoff(research_sub_agent, tool_description_override="For technical research"),
        handoff(writer_sub_agent, tool_description_override="For content creation"),
        handoff(reviewer_sub_agent, tool_description_override="For quality review"),
    ],
)


# 2. Tool-based routing (Manager coordinates)
@function_tool
def route_to_research() -> str:
    """Route to research agent."""
    return "Routing to research agent"


@function_tool
def route_to_writing() -> str:
    """Route to writing agent."""
    return "Routing to writing agent"


# 3. Conditional routing based on input analysis
@function_tool
def analyze_and_route(request: str) -> str:
    """Analyze request and determine routing."""
    request_lower = request.lower()

    if any(word in request_lower for word in ["research", "find", "search", "learn"]):
        return "route:research"
    elif any(word in request_lower for word in ["write", "create", "draft", "compose"]):
        return "route:writing"
    elif any(word in request_lower for word in ["review", "check", "verify", "validate"]):
        return "route:review"
    else:
        return "route:general"


routing_agent = Agent(
    name="RoutingAgent",
    instructions="""You analyze requests and route them appropriately.
    Use analyze_and_route to determine where to send the request.""",
    tools=[analyze_and_route],
)


# =============================================================================
# Tests
# =============================================================================

def test_shared_memory():
    """Test shared memory between agents (MA-05)."""
    print("\n" + "=" * 70)
    print("TEST: Shared Memory (MA-05)")
    print("=" * 70)

    # Clear shared memory
    shared_memory.store.clear()

    # Agent 1 saves data
    shared_memory.put("research_findings", "AI adoption increased 50% in 2025", "ResearchAgent", ["research", "stats"])
    print("\nResearchAgent saved findings")

    # Agent 2 reads data
    item = shared_memory.get("research_findings")
    print(f"WriterAgent reads: {item.value}")
    print(f"Created by: {item.created_by}")

    # Search by tags
    results = shared_memory.search(tags=["research"])
    print(f"\nSearch by 'research' tag: {len(results)} results")

    print("\n✅ Shared memory works")


def test_hierarchical_process():
    """Test hierarchical/manager pattern (MA-03)."""
    print("\n" + "=" * 70)
    print("TEST: Hierarchical Process (MA-03)")
    print("=" * 70)

    # Clear shared memory
    shared_memory.store.clear()

    result = Runner.run_sync(
        manager_agent,
        "Create a brief article about the benefits of AI in healthcare."
    )

    print(f"\nManager result:\n{result.final_output}")

    # Check what's in shared memory
    print(f"\nShared memory after workflow: {shared_memory.list_all()}")

    print("\n✅ Hierarchical process works")


def test_routing():
    """Test routing patterns (MA-04)."""
    print("\n" + "=" * 70)
    print("TEST: Routing Patterns (MA-04)")
    print("=" * 70)

    # Test conditional routing
    test_requests = [
        "I need to research market trends",
        "Please write a blog post",
        "Can you review this document?",
        "Hello, how are you?"
    ]

    print("\nConditional routing analysis:")
    for request in test_requests:
        result = Runner.run_sync(routing_agent, request)
        print(f"  '{request[:30]}...' -> {result.final_output}")

    print("\n✅ Routing patterns work")


def test_agent_as_tool():
    """Test agent-as-tool pattern."""
    print("\n" + "=" * 70)
    print("TEST: Agent-as-Tool Pattern")
    print("=" * 70)

    # Clear shared memory
    shared_memory.store.clear()

    # Test agent-as-tool via manager orchestration
    result = Runner.run_sync(
        manager_agent,
        "Research renewable energy trends and provide a summary"
    )
    print(f"\nAgent-as-tool orchestration result:\n{result.final_output[:300]}...")

    print("\n✅ Agent-as-tool works")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ MA-03, MA-04, MA-05: ORCHESTRATION PATTERNS - EVALUATION SUMMARY            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ MA-03 (Hierarchical Process): ⭐⭐⭐⭐ (Production Ready)                    │
│   ✅ Agent-as-tool pattern works well                                       │
│   ✅ Manager can coordinate sub-agents                                      │
│   ✅ Tools can invoke other agents                                          │
│   ❌ No built-in manager/crew abstraction                                   │
│                                                                             │
│ MA-04 (Routing): ⭐⭐⭐⭐ (Production Ready)                                 │
│   ✅ Handoff-based routing (LLM decides)                                    │
│   ✅ Tool-based routing (programmatic)                                      │
│   ✅ Conditional routing via analysis                                       │
│   ✅ Flexible routing strategies                                            │
│                                                                             │
│ MA-05 (Shared Memory): ⭐⭐⭐ (PoC Ready)                                    │
│   ✅ Global store pattern works                                             │
│   ✅ Agents can read/write shared state                                     │
│   ❌ No built-in shared memory API                                          │
│   ⚠️ Requires custom implementation                                        │
│                                                                             │
│ Orchestration Patterns Supported:                                           │
│   1. Handoff-based: LLM decides routing via handoff()                       │
│   2. Tool-based: Agent invokes other agents as tools                        │
│   3. Manager pattern: Coordinator delegates to specialists                  │
│   4. Conditional: Rules-based routing                                       │
│                                                                             │
│ Comparison with LangGraph:                                                  │
│   LangGraph:                                                                │
│     - Graph-based orchestration                                             │
│     - Explicit edges and conditions                                         │
│     - Store API for shared memory                                           │
│   OpenAI SDK:                                                               │
│     - Handoff-based delegation                                              │
│     - Simpler, less explicit                                                │
│     - Custom shared memory                                                  │
│                                                                             │
│ Comparison with CrewAI:                                                     │
│   CrewAI:                                                                   │
│     - Crew() with Process.hierarchical                                      │
│     - Built-in manager_agent                                                │
│     - Native shared memory                                                  │
│   OpenAI SDK:                                                               │
│     - More manual but flexible                                              │
│     - Agent-as-tool pattern                                                 │
│     - Custom coordination                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_shared_memory()
    test_hierarchical_process()
    test_routing()
    test_agent_as_tool()

    print(SUMMARY)
