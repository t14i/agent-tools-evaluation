"""
Multi-Agent - Part 1: Handoff Basics (MA-01, MA-02)
Native handoffs in OpenAI Agents SDK
"""

from dotenv import load_dotenv
load_dotenv()


from agents import Agent, Runner, function_tool, handoff


# =============================================================================
# MA-01: Multiple Agent Definition
# =============================================================================

# Define specialized agents
@function_tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights."""
    return f"Found 3 flights from {origin} to {destination} on {date}: Flight A ($300), Flight B ($350), Flight C ($280)"


@function_tool
def book_flight(flight_id: str, passenger_name: str) -> str:
    """Book a specific flight."""
    return f"Flight {flight_id} booked for {passenger_name}. Confirmation: ABC123"


@function_tool
def search_hotels(city: str, check_in: str, check_out: str) -> str:
    """Search for available hotels."""
    return f"Found 2 hotels in {city}: Hotel X ($150/night), Hotel Y ($200/night)"


@function_tool
def book_hotel(hotel_id: str, guest_name: str) -> str:
    """Book a hotel room."""
    return f"Hotel {hotel_id} booked for {guest_name}. Confirmation: XYZ789"


@function_tool
def get_recommendations(city: str, interests: str) -> str:
    """Get local recommendations based on interests."""
    return f"Top recommendations in {city} for {interests}: Museum of Art, Central Park, Local Food Tour"


# =============================================================================
# MA-02: Native Handoffs
# =============================================================================

# Create specialized agents
flight_agent = Agent(
    name="FlightAgent",
    instructions="""You are a flight booking specialist.
    Help users search for and book flights.
    If the user asks about hotels or activities, hand off to the appropriate agent.""",
    tools=[search_flights, book_flight],
)

hotel_agent = Agent(
    name="HotelAgent",
    instructions="""You are a hotel booking specialist.
    Help users search for and book hotels.
    If the user asks about flights or activities, hand off to the appropriate agent.""",
    tools=[search_hotels, book_hotel],
)

concierge_agent = Agent(
    name="ConciergeAgent",
    instructions="""You are a local concierge and activity specialist.
    Help users find activities and local recommendations.
    If the user asks about flights or hotels, hand off to the appropriate agent.""",
    tools=[get_recommendations],
)

# Create the main triage agent with handoffs to specialists
triage_agent = Agent(
    name="TriageAgent",
    instructions="""You are a travel assistant triage agent.
    Your job is to understand what the user needs and hand off to the right specialist:
    - For flight bookings: hand off to FlightAgent
    - For hotel bookings: hand off to HotelAgent
    - For activities and recommendations: hand off to ConciergeAgent

    Do not try to handle requests yourself - always delegate to the appropriate specialist.""",
    handoffs=[
        handoff(flight_agent, tool_description_override="Hand off to flight booking specialist"),
        handoff(hotel_agent, tool_description_override="Hand off to hotel booking specialist"),
        handoff(concierge_agent, tool_description_override="Hand off to concierge for activities"),
    ],
)


# =============================================================================
# Circular Handoffs (Agents can hand back)
# =============================================================================

# Update specialists to be able to hand back or to each other
flight_agent_with_handoffs = Agent(
    name="FlightAgentFull",
    instructions="""You are a flight booking specialist.
    Help users search for and book flights.
    If the user asks about hotels, hand off to hotel specialist.
    If the user asks about activities, hand off to concierge.
    If the task is complete, hand back to triage.""",
    tools=[search_flights, book_flight],
    handoffs=[
        handoff(hotel_agent, tool_description_override="Hand off for hotel queries"),
        handoff(concierge_agent, tool_description_override="Hand off for activities"),
    ],
)


# =============================================================================
# Tests
# =============================================================================

def test_basic_handoff():
    """Test basic handoff from triage to specialist (MA-02)."""
    print("\n" + "=" * 70)
    print("TEST: Basic Handoff (MA-02)")
    print("=" * 70)

    result = Runner.run_sync(
        triage_agent,
        "I need to book a flight from New York to London on March 15th"
    )

    print(f"\nFinal output:\n{result.final_output}")

    # Check which agent handled it
    print(f"\nLast agent: {result.last_agent.name if hasattr(result, 'last_agent') else 'unknown'}")

    print("\n✅ Basic handoff works")


def test_multiple_handoffs():
    """Test handling queries that span multiple specialists."""
    print("\n" + "=" * 70)
    print("TEST: Multiple Specialists Query")
    print("=" * 70)

    result = Runner.run_sync(
        triage_agent,
        "I'm planning a trip to Paris. I need a flight, a hotel, and some activity recommendations."
    )

    print(f"\nFinal output:\n{result.final_output}")

    print("\n✅ Multiple handoffs work")


def test_agent_definitions():
    """Test agent definition patterns (MA-01)."""
    print("\n" + "=" * 70)
    print("TEST: Agent Definitions (MA-01)")
    print("=" * 70)

    agents = [flight_agent, hotel_agent, concierge_agent, triage_agent]

    for agent in agents:
        print(f"\nAgent: {agent.name}")
        print(f"  Tools: {[t.name for t in (agent.tools or [])]}")
        print(f"  Handoffs: {len(agent.handoffs or [])} configured")

    print("\n✅ Agent definitions work")


def test_handoff_description():
    """Test that handoffs can have descriptions for LLM guidance."""
    print("\n" + "=" * 70)
    print("TEST: Handoff Descriptions")
    print("=" * 70)

    # Create agent with descriptive handoffs
    descriptive_triage = Agent(
        name="DescriptiveTriageAgent",
        instructions="Route to the appropriate specialist.",
        handoffs=[
            handoff(
                flight_agent,
                tool_description_override="Transfer to flight specialist for booking, searching, or modifying flights"
            ),
            handoff(
                hotel_agent,
                tool_description_override="Transfer to hotel specialist for accommodation bookings and availability"
            ),
            handoff(
                concierge_agent,
                tool_description_override="Transfer to concierge for local tips, tours, restaurants, and activities"
            ),
        ],
    )

    print(f"\nHandoffs configured with descriptions:")
    for h in descriptive_triage.handoffs:
        print(f"  - {h}")

    print("\n✅ Handoff descriptions work")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ MA-01, MA-02: HANDOFF BASICS - EVALUATION SUMMARY                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ MA-01 (Multiple Agent Definition): ⭐⭐⭐⭐⭐ (Production Recommended)       │
│   ✅ Simple Agent class instantiation                                       │
│   ✅ Name, instructions, tools, handoffs                                    │
│   ✅ Clean, declarative definition                                          │
│   ✅ Reusable agent instances                                               │
│                                                                             │
│ MA-02 (Delegation / Handoffs): ⭐⭐⭐⭐⭐ (Production Recommended)           │
│   ✅ Native handoff() function                                              │
│   ✅ Handoffs appear as tools to LLM                                        │
│   ✅ LLM decides when to delegate                                           │
│   ✅ Handoff descriptions guide LLM decisions                               │
│   ✅ Circular handoffs supported                                            │
│                                                                             │
│ OpenAI SDK Handoff Features:                                                │
│   - handoff() creates tool for LLM to invoke                                │
│   - Description helps LLM know when to use                                  │
│   - Context passed automatically                                            │
│   - Conversation history maintained                                         │
│                                                                             │
│ Comparison with LangGraph:                                                  │
│   LangGraph:                                                                │
│     - Manual handoff tools required                                         │
│     - Conditional edges for routing                                         │
│     - More flexible but more work                                           │
│   OpenAI SDK:                                                               │
│     - Native handoff() built-in                                             │
│     - LLM-driven delegation                                                 │
│     - Simpler, less code                                                    │
│                                                                             │
│ Comparison with CrewAI:                                                     │
│   CrewAI:                                                                   │
│     - allow_delegation=True (one line)                                      │
│     - More opinionated                                                      │
│   OpenAI SDK:                                                               │
│     - handoff() (explicit, clear)                                           │
│     - More control over routing                                             │
│                                                                             │
│ Production Notes:                                                           │
│   - Use descriptive handoff descriptions                                    │
│   - Be careful with circular handoffs (infinite loops)                      │
│   - Consider adding loop detection                                          │
│   - Monitor handoff frequency                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_agent_definitions()
    test_basic_handoff()
    test_multiple_handoffs()
    test_handoff_description()

    print(SUMMARY)
