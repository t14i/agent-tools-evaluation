"""
LangGraph Multi-Agent - Supervisor Pattern (MA-01, MA-03, MA-05)
Multiple agents with a supervisor for task delegation.

Evaluation: MA-01 (Multiple Agent Definition), MA-03 (Hierarchical Process), MA-05 (Shared Memory)
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# =============================================================================
# SHARED MEMORY STORE (MA-05)
# =============================================================================

# Shared store for cross-agent memory
shared_store = InMemoryStore()


# =============================================================================
# AGENT TOOLS
# =============================================================================

@tool
def research_topic(query: str) -> str:
    """Research a topic and return findings."""
    # Simulated research
    return f"Research findings for '{query}': Key points include market trends, competitor analysis, and user feedback."


@tool
def write_content(topic: str, style: str = "professional") -> str:
    """Write content on a topic with specified style."""
    return f"Written {style} content about '{topic}': [Generated article with introduction, body, and conclusion]"


@tool
def review_content(content: str) -> str:
    """Review content for quality and accuracy."""
    return f"Review complete: Content is accurate, well-structured, but could use more specific examples."


@tool
def save_to_memory(key: str, value: str, namespace: str = "shared") -> str:
    """Save information to shared memory."""
    shared_store.put((namespace,), key, {"text": value})
    return f"Saved '{key}' to shared memory namespace '{namespace}'"


@tool
def get_from_memory(key: str, namespace: str = "shared") -> str:
    """Retrieve information from shared memory."""
    item = shared_store.get((namespace,), key)
    if item:
        return f"Retrieved: {item.value.get('text', 'No text')}"
    return f"No data found for key '{key}' in namespace '{namespace}'"


# =============================================================================
# AGENT DEFINITIONS (MA-01)
# =============================================================================

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# Research Agent
researcher_tools = [research_topic, save_to_memory, get_from_memory]
researcher_prompt = """You are a Research Agent specialized in gathering information.
Your role is to research topics thoroughly and save important findings to shared memory.
Always save key findings using save_to_memory so other agents can access them."""

# Writer Agent
writer_tools = [write_content, get_from_memory, save_to_memory]
writer_prompt = """You are a Writer Agent specialized in creating content.
Before writing, check shared memory for research findings using get_from_memory.
Use the findings to create well-informed content."""

# Reviewer Agent
reviewer_tools = [review_content, get_from_memory]
reviewer_prompt = """You are a Reviewer Agent specialized in quality assurance.
Review content for accuracy, clarity, and completeness.
Check shared memory for context if needed."""


# =============================================================================
# STATE DEFINITION
# =============================================================================

class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    task_status: dict


# =============================================================================
# SUPERVISOR NODE (MA-03)
# =============================================================================

def supervisor(state: SupervisorState) -> SupervisorState:
    """
    Supervisor node that decides which agent to route to next.
    Implements MA-03: Hierarchical Process.
    """
    messages = state["messages"]
    task_status = state.get("task_status", {})

    # Supervisor prompt
    supervisor_llm = llm.bind_tools([])

    system_prompt = """You are a Supervisor managing a team of agents:
- researcher: Gathers information on topics
- writer: Creates content based on research
- reviewer: Reviews content for quality

Based on the conversation history, decide which agent should act next.
If the task is complete, respond with "COMPLETE".

Current task status:
- Research done: {research_done}
- Writing done: {writing_done}
- Review done: {review_done}

Respond with ONLY one of: researcher, writer, reviewer, or COMPLETE
""".format(
        research_done=task_status.get("research_done", False),
        writing_done=task_status.get("writing_done", False),
        review_done=task_status.get("review_done", False)
    )

    response = supervisor_llm.invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])

    # Parse supervisor decision
    content = response.content.strip().lower()

    if "complete" in content:
        next_agent = "COMPLETE"
    elif "researcher" in content:
        next_agent = "researcher"
        task_status["research_done"] = True
    elif "writer" in content:
        next_agent = "writer"
        task_status["writing_done"] = True
    elif "reviewer" in content:
        next_agent = "reviewer"
        task_status["review_done"] = True
    else:
        # Default to researcher if unclear
        next_agent = "researcher"

    print(f"  [SUPERVISOR] Decision: {next_agent}")

    return {
        "messages": [AIMessage(content=f"[Supervisor] Routing to: {next_agent}")],
        "next_agent": next_agent,
        "task_status": task_status
    }


# =============================================================================
# AGENT NODES
# =============================================================================

def create_agent_node(agent_name: str, tools: list, system_prompt: str):
    """Create an agent node function."""
    agent = create_react_agent(llm, tools)

    def agent_node(state: SupervisorState) -> SupervisorState:
        """Execute the agent."""
        messages = state["messages"]

        # Add system prompt
        agent_messages = [
            SystemMessage(content=system_prompt),
            *messages
        ]

        result = agent.invoke({"messages": agent_messages})
        new_messages = result["messages"]

        # Extract only new messages (after the input)
        output_messages = new_messages[len(agent_messages):]

        print(f"  [{agent_name.upper()}] Generated {len(output_messages)} messages")

        return {
            "messages": output_messages,
            "next_agent": "",
            "task_status": state.get("task_status", {})
        }

    return agent_node


researcher_node = create_agent_node("researcher", researcher_tools, researcher_prompt)
writer_node = create_agent_node("writer", writer_tools, writer_prompt)
reviewer_node = create_agent_node("reviewer", reviewer_tools, reviewer_prompt)


# =============================================================================
# ROUTING
# =============================================================================

def route_to_agent(state: SupervisorState) -> Literal["researcher", "writer", "reviewer", "__end__"]:
    """Route to the next agent based on supervisor decision."""
    next_agent = state.get("next_agent", "")

    if next_agent == "COMPLETE" or not next_agent:
        return "__end__"
    elif next_agent == "researcher":
        return "researcher"
    elif next_agent == "writer":
        return "writer"
    elif next_agent == "reviewer":
        return "reviewer"
    else:
        return "__end__"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_supervisor_graph():
    """Build the supervisor-based multi-agent graph."""
    builder = StateGraph(SupervisorState)

    # Add nodes
    builder.add_node("supervisor", supervisor)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)

    # Add edges
    builder.add_edge(START, "supervisor")

    # Supervisor routes to agents
    builder.add_conditional_edges(
        "supervisor",
        route_to_agent,
        ["researcher", "writer", "reviewer", "__end__"]
    )

    # All agents return to supervisor
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")
    builder.add_edge("reviewer", "supervisor")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# TESTS
# =============================================================================

def test_supervisor_routing():
    """Test supervisor routing decisions."""
    print("\n" + "=" * 70)
    print("TEST: Supervisor Routing (MA-03)")
    print("=" * 70)

    graph = build_supervisor_graph()
    config = {"configurable": {"thread_id": "supervisor-test-1"}}

    # Initial request
    print("\n--- Initial Request ---")
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Create an article about AI trends in 2025")],
            "next_agent": "",
            "task_status": {}
        },
        config=config
    )

    print(f"\nFinal task status: {result.get('task_status', {})}")
    print(f"Total messages: {len(result['messages'])}")


def test_shared_memory():
    """Test shared memory between agents."""
    print("\n" + "=" * 70)
    print("TEST: Shared Memory (MA-05)")
    print("=" * 70)

    # Clear shared store
    shared_store._data.clear()

    # Save from one "agent"
    save_to_memory.invoke({"key": "research_topic", "value": "AI trends include: LLMs, agents, multimodal"})

    # Retrieve from another "agent"
    result = get_from_memory.invoke({"key": "research_topic"})
    print(f"\nRetrieved: {result}")

    # List all items in shared namespace
    items = list(shared_store.search(("shared",)))
    print(f"Items in shared namespace: {len(items)}")
    for item in items:
        print(f"  {item.key}: {item.value}")


def test_full_workflow():
    """Test full multi-agent workflow."""
    print("\n" + "=" * 70)
    print("TEST: Full Multi-Agent Workflow")
    print("=" * 70)

    # Clear shared store
    shared_store._data.clear()

    graph = build_supervisor_graph()
    config = {"configurable": {"thread_id": "full-workflow-1"}}

    print("\n--- Starting Workflow ---")
    print("Task: Research and write about sustainable technology")

    result = graph.invoke(
        {
            "messages": [HumanMessage(
                content="Research sustainable technology trends, write an article about it, and review the content."
            )],
            "next_agent": "",
            "task_status": {}
        },
        config=config
    )

    print(f"\n--- Workflow Complete ---")
    print(f"Task status: {result.get('task_status', {})}")

    # Show final message
    final_messages = [m for m in result["messages"] if hasattr(m, "content")]
    if final_messages:
        print(f"\nFinal output preview: {final_messages[-1].content[:200]}...")

    # Check shared memory
    items = list(shared_store.search(("shared",)))
    print(f"\nShared memory items: {len(items)}")


def test_agent_definitions():
    """Test agent definition structure (MA-01)."""
    print("\n" + "=" * 70)
    print("TEST: Agent Definitions (MA-01)")
    print("=" * 70)

    agents = [
        ("Researcher", researcher_tools, researcher_prompt),
        ("Writer", writer_tools, writer_prompt),
        ("Reviewer", reviewer_tools, reviewer_prompt)
    ]

    for name, tools, prompt in agents:
        print(f"\n{name} Agent:")
        print(f"  Tools: {[t.name for t in tools]}")
        print(f"  Prompt: {prompt[:100]}...")


# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ MA-01, MA-03, MA-05: MULTI-AGENT SUPERVISOR - EVALUATION SUMMARY            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ MA-01 (Multiple Agent Definition): ⭐⭐⭐ (PoC Ready)                        │
│   ✅ create_react_agent() for quick agent creation                          │
│   ✅ Custom agent nodes with tools and prompts                              │
│   ✅ Flexible tool assignment per agent                                     │
│   ❌ No declarative agent configuration (like CrewAI)                       │
│   ❌ No role/goal/backstory pattern                                         │
│                                                                             │
│ MA-03 (Hierarchical Process): ⭐⭐⭐⭐ (Production Ready)                    │
│   ✅ Supervisor pattern implementation                                      │
│   ✅ LLM-based routing decisions                                            │
│   ✅ Task status tracking                                                   │
│   ✅ Conditional edges for agent routing                                    │
│   ❌ No built-in manager abstraction (manual implementation)                │
│                                                                             │
│ MA-05 (Shared Memory): ⭐⭐⭐⭐ (Production Ready)                           │
│   ✅ InMemoryStore with namespace isolation                                 │
│   ✅ Cross-agent memory sharing                                             │
│   ✅ Memory tools for agent access                                          │
│   ✅ PostgresStore for production                                           │
│                                                                             │
│ Comparison with CrewAI:                                                     │
│   - CrewAI: Declarative, role/goal/backstory, allow_delegation=True         │
│   - LangGraph: Programmatic, flexible, more control but more code           │
│                                                                             │
│ Production Considerations:                                                  │
│   - Use PostgresStore for shared memory                                     │
│   - Add supervisor decision logging                                         │
│   - Implement timeout for agent execution                                   │
│   - Add error handling per agent                                            │
│   - Consider async execution for parallel agents                            │
│                                                                             │
│ Overall Rating: ⭐⭐⭐ (PoC Ready)                                           │
│   - Supervisor pattern works well                                           │
│   - More code than CrewAI but more flexible                                 │
│   - Shared memory is solid                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_agent_definitions()
    test_shared_memory()
    test_supervisor_routing()
    # test_full_workflow()  # Uncomment to run full workflow (takes longer)

    print(SUMMARY)
