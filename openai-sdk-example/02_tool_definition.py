"""
Tool Calling - Part 1: Tool Definition Methods (TC-01)
@function_tool decorator, Pydantic schemas, tool_choice
"""

from dotenv import load_dotenv
load_dotenv()


from typing import Annotated
from pydantic import BaseModel, Field
from agents import Agent, Runner, function_tool


# =============================================================================
# Method 1: @function_tool decorator (Simple)
# =============================================================================

@function_tool
def get_weather_simple(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"


# =============================================================================
# Method 2: @function_tool with Annotated (Better docs)
# =============================================================================

@function_tool
def get_weather_typed(
    city: Annotated[str, "The city name to get weather for"],
    unit: Annotated[str, "Temperature unit: celsius or fahrenheit"] = "celsius"
) -> str:
    """Get current weather for a city with specified unit."""
    temp = "22°C" if unit == "celsius" else "72°F"
    return f"Weather in {city}: Sunny, {temp}"


# =============================================================================
# Method 3: Pydantic schema (Full control) - TC-05: Argument Validation
# =============================================================================

class WeatherInput(BaseModel):
    """Input schema for weather tool."""
    city: str = Field(description="The city name to get weather for")
    unit: str = Field(default="celsius", description="Temperature unit: celsius or fahrenheit")
    include_forecast: bool = Field(default=False, description="Include 3-day forecast")


@function_tool
def get_weather_pydantic(
    city: str,
    unit: str = "celsius",
    include_forecast: bool = False
) -> str:
    """Get current weather for a city with full options."""
    temp = "22°C" if unit == "celsius" else "72°F"
    result = f"Weather in {city}: Sunny, {temp}"
    if include_forecast:
        result += "\nForecast: Sun, Mon, Tue - Sunny"
    return result


# Note: OpenAI Agents SDK automatically generates Pydantic schemas from type hints
# The args_schema attribute is generated from the function signature


# =============================================================================
# Method 4: Strict mode for JSON schema validation
# =============================================================================

@function_tool(strict_mode=True)
def get_weather_strict(city: str, unit: str = "celsius") -> str:
    """Get weather with strict JSON schema validation."""
    temp = "22°C" if unit == "celsius" else "72°F"
    return f"Weather in {city}: Sunny, {temp}"


# =============================================================================
# Compare tool definitions
# =============================================================================

if __name__ == "__main__":
    tools = [get_weather_simple, get_weather_typed, get_weather_pydantic, get_weather_strict]

    for t in tools:
        print(f"\n{'='*60}")
        print(f"Tool: {t.name}")
        print(f"Description: {t.description}")
        # OpenAI SDK tools have params_json_schema for the schema
        if hasattr(t, 'params_json_schema'):
            print(f"Params Schema: {t.params_json_schema}")

    print("\n" + "="*60)
    print("SUMMARY - TC-01: Tool Definition")
    print("="*60)
    print("""
| Method              | Pros                          | Cons                    |
|---------------------|-------------------------------|-------------------------|
| @function_tool      | Minimal code                  | No arg descriptions     |
| Annotated hints     | Has descriptions              | Verbose for many args   |
| Pydantic schema     | Full control, validation      | More boilerplate        |
| strict_mode=True    | JSON schema compliance        | Less flexible           |

OpenAI SDK Rating: ⭐⭐⭐⭐⭐ (Production Recommended)
- Automatic Pydantic schema generation from type hints
- Strict mode for JSON schema compliance
- Up to 128 tools per agent
""")
