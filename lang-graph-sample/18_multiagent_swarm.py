"""
LangGraph Multi-Agent - Swarm/Handoff Pattern (MA-02, MA-04)
Agents that can hand off control to each other based on conditions.

Evaluation: MA-02 (Delegation), MA-04 (Routing)
"""

from typing import Annotated, TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# =============================================================================
# STATE DEFINITION
# =============================================================================

class SwarmState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str
    handoff_to: Optional[str]
    context: dict


# =============================================================================
# HANDOFF TOOLS (MA-02)
# =============================================================================

def create_handoff_tool(target_agent: str, description: str):
    """Create a tool that hands off to another agent."""

    @tool
    def handoff() -> str:
        f"""Hand off the conversation to {target_agent}. {description}"""
        return f"HANDOFF:{target_agent}"

    handoff.name = f"handoff_to_{target_agent}"
    handoff.description = f"Hand off to {target_agent}. {description}"
    return handoff


# Create handoff tools for each agent
handoff_to_sales = create_handoff_tool("sales", "Use when customer wants to buy something.")
handoff_to_support = create_handoff_tool("support", "Use when customer has technical issues.")
handoff_to_billing = create_handoff_tool("billing", "Use when customer has payment questions.")
handoff_to_triage = create_handoff_tool("triage", "Use to return to triage for re-routing.")


# =============================================================================
# AGENT-SPECIFIC TOOLS
# =============================================================================

@tool
def check_inventory(product: str) -> str:
    """Check product inventory."""
    return f"Product '{product}' is in stock. 50 units available."


@tool
def create_order(product: str, quantity: int, customer_id: str) -> str:
    """Create a new order."""
    return f"Order created: {quantity}x {product} for customer {customer_id}. Order ID: ORD-12345"


@tool
def check_system_status(service: str) -> str:
    """Check system/service status."""
    return f"Service '{service}' is operational. Uptime: 99.9%"


@tool
def create_ticket(issue: str, priority: str = "medium") -> str:
    """Create a support ticket."""
    return f"Support ticket created: {issue} (Priority: {priority}). Ticket ID: TKT-67890"


@tool
def check_balance(customer_id: str) -> str:
    """Check customer account balance."""
    return f"Customer {customer_id} balance: $1,234.56"


@tool
def process_payment(amount: float, customer_id: str) -> str:
    """Process a payment."""
    return f"Payment of ${amount} processed for customer {customer_id}. Transaction ID: TXN-11111"


# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# Triage Agent - Routes to appropriate specialist
triage_tools = [handoff_to_sales, handoff_to_support, handoff_to_billing]
triage_prompt = """You are a Triage Agent. Your job is to understand the customer's needs
and hand off to the appropriate specialist:
- Sales: For purchases, product inquiries, orders
- Support: For technical issues, bugs, service problems
- Billing: For payments, invoices, account balance

Analyze the customer's message and hand off to the right agent.
Always greet the customer first before handing off."""

# Sales Agent
sales_tools = [check_inventory, create_order, handoff_to_support, handoff_to_billing, handoff_to_triage]
sales_prompt = """You are a Sales Agent. Help customers with:
- Product information and inventory
- Creating orders
- Purchase recommendations

If the customer needs technical support or billing help, hand off to the appropriate agent.
If you can't help, hand off back to triage."""

# Support Agent
support_tools = [check_system_status, create_ticket, handoff_to_sales, handoff_to_billing, handoff_to_triage]
support_prompt = """You are a Support Agent. Help customers with:
- Technical issues and troubleshooting
- System status checks
- Creating support tickets

If the customer needs sales or billing help, hand off to the appropriate agent.
If you can't help, hand off back to triage."""

# Billing Agent
billing_tools = [check_balance, process_payment, handoff_to_sales, handoff_to_support, handoff_to_triage]
billing_prompt = """You are a Billing Agent. Help customers with:
- Account balance inquiries
- Payment processing
- Invoice questions

If the customer needs sales or support help, hand off to the appropriate agent.
If you can't help, hand off back to triage."""


# =============================================================================
# AGENT NODES
# =============================================================================

def create_swarm_agent_node(agent_name: str, tools: list, system_prompt: str):
    """Create a swarm agent node that can hand off to other agents."""
    agent = create_react_agent(llm, tools)

    def agent_node(state: SwarmState) -> SwarmState:
        """Execute the agent and check for handoffs."""
        messages = state["messages"]

        # Add system prompt and context
        context = state.get("context", {})
        context_str = f"\nCustomer context: {context}" if context else ""

        agent_messages = [
            SystemMessage(content=system_prompt + context_str),
            *messages
        ]

        result = agent.invoke({"messages": agent_messages})
        new_messages = result["messages"]

        # Extract only new messages
        output_messages = new_messages[len(agent_messages):]

        # Check for handoff in tool responses
        handoff_to = None
        for msg in output_messages:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                if msg.content.startswith("HANDOFF:"):
                    handoff_to = msg.content.split(":")[1]
                    print(f"  [{agent_name.upper()}] Handing off to: {handoff_to}")
                    break

        print(f"  [{agent_name.upper()}] Generated {len(output_messages)} messages")

        return {
            "messages": output_messages,
            "current_agent": agent_name,
            "handoff_to": handoff_to,
            "context": context
        }

    return agent_node


triage_node = create_swarm_agent_node("triage", triage_tools, triage_prompt)
sales_node = create_swarm_agent_node("sales", sales_tools, sales_prompt)
support_node = create_swarm_agent_node("support", support_tools, support_prompt)
billing_node = create_swarm_agent_node("billing", billing_tools, billing_prompt)


# =============================================================================
# ROUTING (MA-04)
# =============================================================================

def route_after_agent(state: SwarmState) -> Literal["triage", "sales", "support", "billing", "__end__"]:
    """Route based on handoff decision."""
    handoff_to = state.get("handoff_to")

    if handoff_to:
        print(f"  [ROUTER] Routing to: {handoff_to}")
        if handoff_to == "triage":
            return "triage"
        elif handoff_to == "sales":
            return "sales"
        elif handoff_to == "support":
            return "support"
        elif handoff_to == "billing":
            return "billing"

    # No handoff - conversation complete
    print("  [ROUTER] No handoff - ending")
    return "__end__"


# =============================================================================
# CONDITIONAL ROUTING (MA-04)
# =============================================================================

def intent_based_router(state: SwarmState) -> Literal["triage", "sales", "support", "billing"]:
    """
    Route based on detected intent in the message.
    This demonstrates MA-04: Routing based on conditions.
    """
    messages = state["messages"]
    if not messages:
        return "triage"

    last_message = messages[-1]
    content = last_message.content.lower() if hasattr(last_message, "content") else ""

    # Simple keyword-based routing (production would use NLU)
    if any(word in content for word in ["buy", "purchase", "order", "price", "inventory"]):
        print("  [INTENT] Detected: sales")
        return "sales"
    elif any(word in content for word in ["error", "bug", "broken", "not working", "issue", "problem"]):
        print("  [INTENT] Detected: support")
        return "support"
    elif any(word in content for word in ["pay", "invoice", "balance", "charge", "refund"]):
        print("  [INTENT] Detected: billing")
        return "billing"
    else:
        print("  [INTENT] Detected: triage (default)")
        return "triage"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_swarm_graph():
    """Build the swarm/handoff multi-agent graph."""
    builder = StateGraph(SwarmState)

    # Add nodes
    builder.add_node("triage", triage_node)
    builder.add_node("sales", sales_node)
    builder.add_node("support", support_node)
    builder.add_node("billing", billing_node)

    # Start with intent-based routing
    builder.add_conditional_edges(
        START,
        intent_based_router,
        ["triage", "sales", "support", "billing"]
    )

    # Each agent can hand off to others
    for agent in ["triage", "sales", "support", "billing"]:
        builder.add_conditional_edges(
            agent,
            route_after_agent,
            ["triage", "sales", "support", "billing", "__end__"]
        )

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# TESTS
# =============================================================================

def test_intent_routing():
    """Test intent-based routing (MA-04)."""
    print("\n" + "=" * 70)
    print("TEST: Intent-Based Routing (MA-04)")
    print("=" * 70)

    graph = build_swarm_graph()

    # Test different intents
    test_cases = [
        ("I want to buy a laptop", "sales"),
        ("My account is showing wrong balance", "billing"),
        ("The website is giving me an error", "support"),
        ("Hello, I need help", "triage"),
    ]

    for message, expected in test_cases:
        print(f"\n--- Message: '{message}' ---")
        config = {"configurable": {"thread_id": f"intent-test-{hash(message)}"}}

        result = graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "current_agent": "",
                "handoff_to": None,
                "context": {}
            },
            config=config
        )

        actual = result.get("current_agent", "unknown")
        status = "✅" if actual == expected else "⚠️"
        print(f"{status} Expected: {expected}, Got: {actual}")


def test_handoff_chain():
    """Test agent-to-agent handoffs (MA-02)."""
    print("\n" + "=" * 70)
    print("TEST: Agent Handoff Chain (MA-02)")
    print("=" * 70)

    graph = build_swarm_graph()
    config = {"configurable": {"thread_id": "handoff-test-1"}}

    # Start with a complex query that requires handoff
    print("\n--- Complex Query Requiring Handoffs ---")
    result = graph.invoke(
        {
            "messages": [HumanMessage(
                content="I bought a product last week but it's not working. Can you help me and also check my account balance?"
            )],
            "current_agent": "",
            "handoff_to": None,
            "context": {"customer_id": "CUST-001"}
        },
        config=config
    )

    print(f"\nFinal agent: {result.get('current_agent')}")
    print(f"Total messages: {len(result['messages'])}")


def test_context_passing():
    """Test context preservation across handoffs."""
    print("\n" + "=" * 70)
    print("TEST: Context Preservation")
    print("=" * 70)

    graph = build_swarm_graph()
    config = {"configurable": {"thread_id": "context-test-1"}}

    # Provide customer context
    context = {
        "customer_id": "CUST-123",
        "membership": "gold",
        "previous_issues": ["shipping delay", "wrong item"]
    }

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="I need to return an item")],
            "current_agent": "",
            "handoff_to": None,
            "context": context
        },
        config=config
    )

    # Context should be preserved
    final_context = result.get("context", {})
    print(f"Context preserved: {final_context == context}")


def test_handoff_tools():
    """Test handoff tool definitions."""
    print("\n" + "=" * 70)
    print("TEST: Handoff Tool Definitions")
    print("=" * 70)

    handoff_tools = [handoff_to_sales, handoff_to_support, handoff_to_billing, handoff_to_triage]

    for t in handoff_tools:
        print(f"\nTool: {t.name}")
        print(f"  Description: {t.description}")

        # Test invocation
        result = t.invoke({})
        print(f"  Returns: {result}")


# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ MA-02 & MA-04: SWARM/HANDOFF PATTERN - EVALUATION SUMMARY                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ MA-02 (Delegation): ⭐⭐⭐ (PoC Ready)                                       │
│   ✅ Handoff via tool calls                                                 │
│   ✅ Agent-to-agent delegation                                              │
│   ✅ Context preservation across handoffs                                   │
│   ❌ No native delegation (like CrewAI's allow_delegation=True)             │
│   ❌ Manual handoff tool creation required                                  │
│                                                                             │
│ MA-04 (Routing): ⭐⭐⭐⭐ (Production Ready)                                 │
│   ✅ Conditional edges for routing                                          │
│   ✅ Intent-based routing                                                   │
│   ✅ Dynamic routing based on state                                         │
│   ✅ Flexible routing logic (keywords, NLU, rules)                          │
│                                                                             │
│ Swarm Pattern Benefits:                                                     │
│   - Agents can naturally hand off mid-conversation                          │
│   - No central supervisor required                                          │
│   - Flexible agent-to-agent communication                                   │
│   - Good for customer service, helpdesk scenarios                           │
│                                                                             │
│ Comparison with CrewAI:                                                     │
│   - CrewAI: allow_delegation=True is one line                               │
│   - LangGraph: Manual handoff tools + routing logic                         │
│   - LangGraph is more flexible but requires more code                       │
│                                                                             │
│ Production Considerations:                                                  │
│   - Use NLU/classifier for intent routing                                   │
│   - Add handoff logging for analytics                                       │
│   - Implement max handoff limit to prevent loops                            │
│   - Add timeout per agent                                                   │
│   - Consider async handoffs for non-blocking                                │
│                                                                             │
│ Overall Rating: ⭐⭐⭐ (PoC Ready)                                           │
│   - Routing is solid and flexible                                           │
│   - Delegation requires manual implementation                               │
│   - Works well but more verbose than CrewAI                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_handoff_tools()
    test_intent_routing()
    test_handoff_chain()
    test_context_passing()

    print(SUMMARY)
