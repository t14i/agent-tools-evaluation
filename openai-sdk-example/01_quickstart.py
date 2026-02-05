"""
OpenAI Agents SDK Quick Start - Minimal configuration
Agent definition -> Tool binding -> Runner execution
"""

from dotenv import load_dotenv
load_dotenv()

from agents import Agent, Runner, function_tool


# 1. Define a simple tool
@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"


# 2. Create an agent
agent = Agent(
    name="WeatherBot",
    instructions="You are a helpful weather assistant. Use the get_weather tool when asked about weather.",
    tools=[get_weather],
)


# 3. Execute with Runner
if __name__ == "__main__":
    result = Runner.run_sync(agent, "What's the weather in Tokyo?")

    print("=== Result ===")
    print(f"Final output: {result.final_output}")

    print("\n=== Messages ===")
    for item in result.new_items:
        print(f"  {item}")


# =============================================================================
# Key Concepts
# =============================================================================
"""
| Element | Description |
|---------|-------------|
| Agent | Core abstraction with name, instructions, tools |
| @function_tool | Decorator for tool definition |
| Runner.run_sync | Synchronous execution |
| result.final_output | Agent's final response |
| result.new_items | All generated items (messages, tool calls, etc.) |

Comparison with LangGraph:
- LangGraph: StateGraph, nodes, edges, compile
- OpenAI SDK: Agent class, Runner execution (simpler)
"""
