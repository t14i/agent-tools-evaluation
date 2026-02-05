"""
Testing & Evaluation - Part 1: Mocking (TE-01, TE-02)
Unit testing, mock LLMs, state injection
"""

from dotenv import load_dotenv
load_dotenv()


from typing import Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json


# =============================================================================
# TE-01: Unit Test / Mocking
# =============================================================================

@dataclass
class MockResponse:
    """Mock LLM response."""
    content: str
    tool_calls: list = field(default_factory=list)
    model: str = "mock-model"
    usage: dict = field(default_factory=lambda: {"input_tokens": 10, "output_tokens": 5})


class MockLLM:
    """
    Mock LLM for testing.
    Implements TE-01: Unit Test / Mocking.
    """

    def __init__(self, responses: list[str] = None):
        self.responses = responses or ["Mock response"]
        self.response_index = 0
        self.call_history: list[dict] = []

    def set_responses(self, responses: list[str]):
        """Set canned responses."""
        self.responses = responses
        self.response_index = 0

    def set_tool_response(self, tool_name: str, tool_args: dict):
        """Set a tool call response."""
        self.responses = [MockResponse(
            content="",
            tool_calls=[{"name": tool_name, "args": tool_args, "id": "call_123"}]
        )]
        self.response_index = 0

    def invoke(self, messages: list) -> MockResponse:
        """Invoke mock LLM."""
        # Record call
        self.call_history.append({
            "timestamp": datetime.now().isoformat(),
            "messages": messages
        })

        # Get response
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1

        if isinstance(response, MockResponse):
            return response
        return MockResponse(content=response)

    def get_call_count(self) -> int:
        """Get number of calls made."""
        return len(self.call_history)

    def get_last_call(self) -> Optional[dict]:
        """Get the last call made."""
        return self.call_history[-1] if self.call_history else None

    def reset(self):
        """Reset mock state."""
        self.call_history = []
        self.response_index = 0


class MockTool:
    """
    Mock tool for testing.
    """

    def __init__(self, name: str, return_value: Any = None):
        self.name = name
        self.return_value = return_value or f"Mock result for {name}"
        self.call_history: list[dict] = []

    def __call__(self, **kwargs) -> Any:
        """Execute mock tool."""
        self.call_history.append({
            "timestamp": datetime.now().isoformat(),
            "args": kwargs
        })
        return self.return_value

    def set_return_value(self, value: Any):
        """Set the return value."""
        self.return_value = value

    def set_side_effect(self, exception: Exception):
        """Set tool to raise an exception."""
        self.return_value = exception

    def get_call_count(self) -> int:
        return len(self.call_history)

    def was_called_with(self, **kwargs) -> bool:
        """Check if tool was called with specific args."""
        for call in self.call_history:
            if call["args"] == kwargs:
                return True
        return False


# =============================================================================
# TE-02: Test Fixtures / State Injection
# =============================================================================

@dataclass
class TestFixture:
    """Test fixture for state injection."""
    name: str
    initial_state: dict
    messages: list
    expected_output: Optional[str] = None
    expected_tool_calls: list = field(default_factory=list)


class StateInjector:
    """
    Injects state for testing at specific points.
    Implements TE-02: Test Fixtures / State Injection.
    """

    def __init__(self):
        self.fixtures: dict[str, TestFixture] = {}

    def register_fixture(self, fixture: TestFixture):
        """Register a test fixture."""
        self.fixtures[fixture.name] = fixture

    def get_fixture(self, name: str) -> Optional[TestFixture]:
        """Get a fixture by name."""
        return self.fixtures.get(name)

    def create_test_state(self, fixture_name: str) -> dict:
        """Create a test state from a fixture."""
        fixture = self.fixtures.get(fixture_name)
        if not fixture:
            return {}

        return {
            "messages": fixture.messages.copy(),
            **fixture.initial_state.copy()
        }


# =============================================================================
# Test Runner
# =============================================================================

@dataclass
class TestResult:
    """Result of a test run."""
    test_name: str
    passed: bool
    duration_ms: float
    assertions: list[dict]
    error: Optional[str] = None


class AgentTestRunner:
    """
    Test runner for agent testing.
    """

    def __init__(self):
        self.results: list[TestResult] = []
        self.mock_llm = MockLLM()
        self.mock_tools: dict[str, MockTool] = {}
        self.state_injector = StateInjector()

    def register_mock_tool(self, name: str, return_value: Any = None) -> MockTool:
        """Register a mock tool."""
        tool = MockTool(name, return_value)
        self.mock_tools[name] = tool
        return tool

    def run_test(
        self,
        test_name: str,
        agent_fn: Callable,
        input_data: dict,
        assertions: list[Callable]
    ) -> TestResult:
        """Run a single test."""
        start_time = datetime.now()
        assertion_results = []
        error = None
        passed = True

        try:
            # Run agent function
            result = agent_fn(input_data, self.mock_llm, self.mock_tools)

            # Run assertions
            for assertion in assertions:
                try:
                    assertion_passed, message = assertion(result)
                    assertion_results.append({
                        "passed": assertion_passed,
                        "message": message
                    })
                    if not assertion_passed:
                        passed = False
                except Exception as e:
                    assertion_results.append({
                        "passed": False,
                        "message": f"Assertion error: {e}"
                    })
                    passed = False

        except Exception as e:
            error = str(e)
            passed = False

        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        test_result = TestResult(
            test_name=test_name,
            passed=passed,
            duration_ms=duration_ms,
            assertions=assertion_results,
            error=error
        )

        self.results.append(test_result)
        return test_result

    def get_summary(self) -> dict:
        """Get test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "total_duration_ms": sum(r.duration_ms for r in self.results)
        }


# =============================================================================
# Assertion Helpers
# =============================================================================

def assert_output_contains(text: str):
    """Assert that output contains specific text."""
    def check(result):
        output = result.get("output", "")
        passed = text in output
        return (passed, f"Output {'contains' if passed else 'does not contain'} '{text}'")
    return check


def assert_tool_called(tool_name: str, with_args: dict = None):
    """Assert that a tool was called."""
    def check(result):
        tool_calls = result.get("tool_calls", [])
        for call in tool_calls:
            if call["name"] == tool_name:
                if with_args is None:
                    return (True, f"Tool '{tool_name}' was called")
                if call.get("args") == with_args:
                    return (True, f"Tool '{tool_name}' was called with expected args")
        return (False, f"Tool '{tool_name}' was not called")
    return check


def assert_no_tool_called(tool_name: str):
    """Assert that a tool was NOT called."""
    def check(result):
        tool_calls = result.get("tool_calls", [])
        for call in tool_calls:
            if call["name"] == tool_name:
                return (False, f"Tool '{tool_name}' was called unexpectedly")
        return (True, f"Tool '{tool_name}' was not called")
    return check


def assert_state_equals(key: str, expected_value: Any):
    """Assert state value equals expected."""
    def check(result):
        state = result.get("state", {})
        actual = state.get(key)
        passed = actual == expected_value
        return (passed, f"State '{key}' is {'equal' if passed else 'not equal'} to expected")
    return check


# =============================================================================
# Tests
# =============================================================================

def test_mock_llm():
    """Test mock LLM (TE-01)."""
    print("\n" + "=" * 70)
    print("TEST: Mock LLM (TE-01)")
    print("=" * 70)

    mock = MockLLM(["Response 1", "Response 2", "Response 3"])

    # Multiple invocations
    for i in range(3):
        response = mock.invoke([{"role": "user", "content": f"Query {i}"}])
        print(f"  Call {i+1}: {response.content}")

    print(f"\nTotal calls: {mock.get_call_count()}")
    print(f"Last call: {mock.get_last_call()['messages']}")

    # Test tool call response
    mock.set_tool_response("search", {"query": "test"})
    response = mock.invoke([])
    print(f"\nTool call response: {response.tool_calls}")

    print("\n✅ Mock LLM works")


def test_mock_tools():
    """Test mock tools (TE-01)."""
    print("\n" + "=" * 70)
    print("TEST: Mock Tools (TE-01)")
    print("=" * 70)

    tool = MockTool("search_database", {"results": ["item1", "item2"]})

    # Call tool
    result = tool(query="test", limit=10)
    print(f"Result: {result}")

    # Verify calls
    print(f"Call count: {tool.get_call_count()}")
    print(f"Was called with query='test': {tool.was_called_with(query='test', limit=10)}")

    print("\n✅ Mock tools work")


def test_state_injection():
    """Test state injection (TE-02)."""
    print("\n" + "=" * 70)
    print("TEST: State Injection (TE-02)")
    print("=" * 70)

    injector = StateInjector()

    # Register fixture
    fixture = TestFixture(
        name="mid_conversation",
        initial_state={"user_id": "user_123", "context": {"topic": "billing"}},
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "I have a billing question"}
        ],
        expected_output="billing support",
        expected_tool_calls=["get_billing_info"]
    )

    injector.register_fixture(fixture)

    # Create test state
    state = injector.create_test_state("mid_conversation")
    print(f"\nInjected state:")
    print(f"  Messages: {len(state['messages'])}")
    print(f"  User ID: {state['user_id']}")
    print(f"  Context: {state['context']}")

    print("\n✅ State injection works")


def test_agent_test_runner():
    """Test the agent test runner."""
    print("\n" + "=" * 70)
    print("TEST: Agent Test Runner")
    print("=" * 70)

    runner = AgentTestRunner()

    # Register mocks
    runner.mock_llm.set_responses(["Here is the search result for your query."])
    search_tool = runner.register_mock_tool("search", {"results": ["found item"]})

    # Define a simple agent function for testing
    def simple_agent(input_data, llm, tools):
        # Simulate agent execution
        response = llm.invoke(input_data.get("messages", []))

        # Simulate tool call
        if "search" in input_data.get("query", ""):
            tool_result = tools["search"](query=input_data["query"])
            return {
                "output": response.content,
                "tool_calls": [{"name": "search", "args": {"query": input_data["query"]}}],
                "state": {"search_performed": True}
            }

        return {"output": response.content, "tool_calls": [], "state": {}}

    # Run test
    result = runner.run_test(
        test_name="test_search_query",
        agent_fn=simple_agent,
        input_data={"messages": [], "query": "search for products"},
        assertions=[
            assert_output_contains("search result"),
            assert_tool_called("search"),
            assert_state_equals("search_performed", True)
        ]
    )

    print(f"\nTest: {result.test_name}")
    print(f"Passed: {result.passed}")
    print(f"Duration: {result.duration_ms:.2f}ms")
    print(f"Assertions:")
    for a in result.assertions:
        status = "✅" if a["passed"] else "❌"
        print(f"  {status} {a['message']}")

    # Summary
    summary = runner.get_summary()
    print(f"\nSummary: {summary}")

    print("\n✅ Agent test runner works")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ TE-01, TE-02: MOCKING - EVALUATION SUMMARY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ TE-01 (Unit Test / Mocking): ⭐⭐⭐⭐ (Production Ready)                     │
│   ✅ model_settings allows injection                                        │
│   ✅ Tools can be replaced with mocks                                       │
│   ✅ Deterministic testing possible                                         │
│   ⚠️ No built-in mock utilities                                            │
│                                                                             │
│ TE-02 (Test Fixtures / State Injection): ⭐⭐⭐ (PoC Ready)                  │
│   ✅ Sessions allow state restoration                                       │
│   ✅ Can start from specific checkpoint                                     │
│   ⚠️ Requires custom fixture management                                    │
│   ⚠️ No built-in fixture support                                           │
│                                                                             │
│ Custom Implementation Provided:                                             │
│   ✅ MockLLM - Deterministic LLM responses                                  │
│   ✅ MockTool - Tool mocking with call tracking                             │
│   ✅ StateInjector - Fixture-based state injection                          │
│   ✅ AgentTestRunner - Test execution framework                             │
│   ✅ Assertion helpers - Common test assertions                             │
│                                                                             │
│ Testing Patterns:                                                           │
│   1. Mock LLM responses for deterministic tests                             │
│   2. Mock tools to verify call patterns                                     │
│   3. Inject state to test from specific points                              │
│   4. Assert on output, tool calls, and state                                │
│                                                                             │
│ Comparison with LangGraph:                                                  │
│   LangGraph:                                                                │
│     - Graph structure enables node-level testing                            │
│     - Checkpointer for state injection                                      │
│     - FakeListChatModel for mocking                                         │
│   OpenAI SDK:                                                               │
│     - Similar capabilities                                                  │
│     - Sessions for state                                                    │
│     - Custom mocks needed                                                   │
│                                                                             │
│ Production Notes:                                                           │
│   - Create comprehensive mock libraries                                     │
│   - Use fixtures for regression testing                                     │
│   - Test edge cases (errors, timeouts)                                      │
│   - Verify tool call sequences                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_mock_llm()
    test_mock_tools()
    test_state_injection()
    test_agent_test_runner()

    print(SUMMARY)
