"""
01_quickstart.py - CrewAI Quick Start

Purpose: Understanding CrewAI's basic structure
- Minimal configuration: 1 Agent + 1 Task + 1 Crew
- Verify effect of role/goal/backstory
- Python-based definition

LangGraph Comparison:
- CrewAI: Declarative definition with Agent/Task/Crew
- LangGraph: Explicit connections with StateGraph/Node/Edge
"""

from crewai import Agent, Task, Crew


def main():
    # Agent definition
    # role: Agent's role (included in system prompt to LLM)
    # goal: Agent's objective (guidance for task execution)
    # backstory: Background setting (more detailed context)
    researcher = Agent(
        role="Research Analyst",
        goal="Provide accurate and insightful analysis on given topics",
        backstory="""You are an experienced research analyst with a keen eye
        for detail. You excel at gathering information and presenting it
        in a clear, structured manner.""",
        verbose=True,  # Display execution process
    )

    # Task definition
    # description: Detailed description of the task
    # expected_output: Expected output format
    # agent: Agent that executes this task
    research_task = Task(
        description="""Research the current state of AI agent frameworks.
        Focus on:
        1. Main frameworks available
        2. Key differences between them
        3. Typical use cases""",
        expected_output="""A structured report containing:
        - List of major AI agent frameworks
        - Comparison table of features
        - Recommendations for different use cases""",
        agent=researcher,
    )

    # Crew definition
    # agents: List of participating agents
    # tasks: List of tasks to execute
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True,  # Detailed log output
    )

    # Execute
    print("=" * 60)
    print("CrewAI Quick Start - Minimal Configuration")
    print("=" * 60)

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    print(result)

    # Check CrewOutput attributes
    print("\n" + "=" * 60)
    print("CrewOutput attributes:")
    print("=" * 60)
    print(f"Type: {type(result)}")
    print(f"Raw output: {result.raw[:200]}..." if len(result.raw) > 200 else f"Raw output: {result.raw}")

    if result.tasks_output:
        print(f"\nTasks output count: {len(result.tasks_output)}")
        for i, task_output in enumerate(result.tasks_output):
            print(f"  Task {i+1}: {type(task_output)}")

    # Token usage (if available)
    if hasattr(result, "token_usage"):
        print(f"\nToken usage: {result.token_usage}")


if __name__ == "__main__":
    main()
