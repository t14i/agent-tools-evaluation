"""
05_hitl_task_input.py - Task-level Human-in-the-Loop

Purpose: Verify HITL functionality with human_input=True
- Behavior of human_input=True
- Verify waiting state for input
- Continuation after feedback

LangGraph Comparison:
- LangGraph: interrupt() allows interruption at any point
- CrewAI: human_input flag at task level (simpler)
"""

from crewai import Agent, Task, Crew
from crewai.tools import tool


@tool("Draft Generator")
def generate_draft(topic: str) -> str:
    """
    Generate a draft document on a given topic.

    Args:
        topic: The topic to write about
    """
    return f"""
Draft Document: {topic}
========================

Introduction:
This document covers the key aspects of {topic}.

Main Points:
1. First important point about {topic}
2. Second consideration for {topic}
3. Third aspect of {topic}

Conclusion:
In summary, {topic} is a significant area that requires attention.

[This is an AI-generated draft for review]
"""


def main():
    print("=" * 60)
    print("HITL: Task-level human_input Test")
    print("=" * 60)
    print("""
This example demonstrates human_input=True on a Task.
When enabled, the agent will pause and ask for human feedback
before finalizing the task output.

LangGraph Comparison:
- CrewAI human_input: Automatic pause at task completion
- LangGraph interrupt(): Manual placement, more flexible
""")

    # Writer agent
    writer = Agent(
        role="Content Writer",
        goal="Create high-quality content based on user requirements",
        backstory="""You are a professional content writer who values
        collaboration. You create drafts and refine them based on feedback.""",
        tools=[generate_draft],
        verbose=True,
    )

    # Editor agent for review
    editor = Agent(
        role="Content Editor",
        goal="Review and improve content quality",
        backstory="""You are a meticulous editor who ensures content
        meets high standards before publication.""",
        verbose=True,
    )

    # Task WITH human_input - will pause for feedback
    writing_task = Task(
        description="""Create a draft document about "Best Practices for AI Agent Development".
        Use the draft generator tool to create an initial draft.
        The document should be professional and informative.""",
        expected_output="A well-structured draft document ready for review",
        agent=writer,
        human_input=True,  # <-- This enables HITL
    )

    # Follow-up task
    editing_task = Task(
        description="""Review and improve the draft from the writing task.
        Incorporate any feedback provided during the review.
        Ensure the final version is polished and professional.""",
        expected_output="A final, polished document incorporating all feedback",
        agent=editor,
    )

    crew = Crew(
        agents=[writer, editor],
        tasks=[writing_task, editing_task],
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Starting Crew Execution")
    print("=" * 60)
    print("""
NOTE: When the writing task completes, you will be prompted
for feedback. This is the human_input=True feature in action.

Type your feedback when prompted, or press Enter to accept.
""")

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("Final Result:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
