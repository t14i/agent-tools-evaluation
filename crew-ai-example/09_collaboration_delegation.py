"""
09_collaboration_delegation.py - Inter-Agent Delegation

Purpose: Verify CrewAI's agent delegation feature
- Behavior of allow_delegation=True
- Delegation trigger conditions
- Delegation target selection logic

CrewAI-specific feature:
- LangGraph has no built-in delegation feature
- LangGraph requires explicit routing implementation
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool


# =============================================================================
# Tools for specialized agents
# =============================================================================
@tool("Code Analyzer")
def analyze_code(code_snippet: str) -> str:
    """
    Analyze a code snippet for quality and issues.

    Args:
        code_snippet: The code to analyze
    """
    return f"""
Code Analysis Report:
- Lines of code: {len(code_snippet.splitlines())}
- Complexity: Medium
- Suggestions:
  1. Consider adding type hints
  2. Add docstrings for documentation
  3. Consider error handling
"""


@tool("Security Scanner")
def security_scan(code_snippet: str) -> str:
    """
    Scan code for security vulnerabilities.

    Args:
        code_snippet: The code to scan
    """
    return f"""
Security Scan Report:
- Vulnerabilities found: 0 critical, 1 warning
- Warning: Input validation recommended
- No SQL injection risks detected
- No hardcoded credentials found
"""


@tool("Performance Profiler")
def profile_performance(code_snippet: str) -> str:
    """
    Profile code for performance issues.

    Args:
        code_snippet: The code to profile
    """
    return f"""
Performance Profile:
- Time complexity: O(n)
- Space complexity: O(1)
- Suggestions:
  1. Consider caching for repeated calls
  2. Use generators for large datasets
"""


def main():
    print("=" * 60)
    print("Role-Based Collaboration: Delegation Test")
    print("=" * 60)
    print("""
This example demonstrates CrewAI's agent delegation feature.
When allow_delegation=True, agents can delegate tasks to other
agents they deem more suitable.

Key Points:
- Delegation is automatic based on agent roles/capabilities
- The delegating agent decides when another agent is better suited
- This is a CrewAI-specific feature not found in LangGraph

LangGraph Comparison:
- CrewAI: allow_delegation=True (automatic)
- LangGraph: Explicit router/conditional edges (manual)
""")

    # ==========================================================================
    # Define specialized agents
    # ==========================================================================

    # Lead developer who can delegate
    lead_developer = Agent(
        role="Lead Developer",
        goal="Oversee code reviews and delegate to specialists when needed",
        backstory="""You are a senior lead developer with broad knowledge.
        You can handle general tasks but prefer to delegate specialized
        work to experts. You know when security or performance reviews
        need a specialist's attention.""",
        tools=[analyze_code],
        allow_delegation=True,  # <-- Enable delegation
        verbose=True,
    )

    # Security specialist
    security_expert = Agent(
        role="Security Expert",
        goal="Identify and address security vulnerabilities in code",
        backstory="""You are a cybersecurity specialist focused on
        application security. You excel at finding vulnerabilities
        and recommending secure coding practices.""",
        tools=[security_scan],
        allow_delegation=False,  # Specialists don't delegate
        verbose=True,
    )

    # Performance specialist
    performance_expert = Agent(
        role="Performance Engineer",
        goal="Optimize code for maximum performance",
        backstory="""You are a performance optimization specialist.
        You analyze code for efficiency and provide optimization
        recommendations.""",
        tools=[profile_performance],
        allow_delegation=False,
        verbose=True,
    )

    # ==========================================================================
    # Define tasks
    # ==========================================================================

    # Main review task assigned to lead
    comprehensive_review = Task(
        description="""Perform a comprehensive review of the following code:

```python
def process_user_data(user_input):
    # Process user data from form submission
    data = eval(user_input)  # Parse the input
    results = []
    for item in data:
        results.append(item * 2)
    return results
```

The review should cover:
1. General code quality
2. Security considerations (delegate to security expert if needed)
3. Performance analysis (delegate to performance expert if needed)

Provide a unified report combining all findings.""",
        expected_output="""A comprehensive code review report including:
- Code quality assessment
- Security findings
- Performance analysis
- Overall recommendations""",
        agent=lead_developer,
    )

    # ==========================================================================
    # Create crew with all agents
    # ==========================================================================
    crew = Crew(
        agents=[lead_developer, security_expert, performance_expert],
        tasks=[comprehensive_review],
        process=Process.sequential,  # Sequential for clear delegation observation
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Executing Crew with Delegation")
    print("=" * 60)
    print("""
Watch for delegation behavior:
- Lead developer may delegate security review to Security Expert
- Lead developer may delegate performance analysis to Performance Engineer
""")

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    print(result)

    # Show delegation metrics if available
    if hasattr(result, "token_usage"):
        print("\n" + "=" * 60)
        print("Token Usage:")
        print("=" * 60)
        print(result.token_usage)


def demo_no_delegation():
    """Demonstrate behavior without delegation for comparison."""
    print("\n" + "=" * 60)
    print("Comparison: Without Delegation")
    print("=" * 60)

    solo_developer = Agent(
        role="Solo Developer",
        goal="Review code independently without delegation",
        backstory="You must handle all aspects of code review yourself.",
        tools=[analyze_code, security_scan, profile_performance],
        allow_delegation=False,  # No delegation
        verbose=True,
    )

    solo_review = Task(
        description="""Review this code for quality, security, and performance:

```python
def calculate_total(items):
    return sum(item.price for item in items)
```""",
        expected_output="A brief code review covering all aspects",
        agent=solo_developer,
    )

    crew = Crew(
        agents=[solo_developer],
        tasks=[solo_review],
        verbose=True,
    )

    result = crew.kickoff()
    print("\nSolo Review Result:")
    print(result)


if __name__ == "__main__":
    main()
    # Uncomment to see comparison without delegation:
    # demo_no_delegation()
