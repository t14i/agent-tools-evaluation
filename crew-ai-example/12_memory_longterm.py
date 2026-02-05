"""
12_memory_longterm.py - Long-term Memory

Purpose: Verify CrewAI's long-term memory feature
- Long-term memory persistence
- Memory retention across sessions
- Effect of entity memory

Usage:
1. First run: python 12_memory_longterm.py --session 1
2. Second run: python 12_memory_longterm.py --session 2

LangGraph Comparison:
- LangGraph: Persist memory with Checkpointer
- CrewAI: Automatic persistence with memory=True + long_term_memory=True
"""

import argparse
from pathlib import Path

from crewai import Agent, Task, Crew


# Memory storage location (CrewAI default uses SQLite)
MEMORY_DIR = Path("./db/memory")
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def check_memory_storage():
    """Check what's stored in memory."""
    print("\n[Memory Storage Check]")

    # Check possible memory locations
    locations = [
        Path.home() / ".crewai" / "memory",
        Path("./db"),
        Path("./.crewai"),
        MEMORY_DIR,
    ]

    for loc in locations:
        if loc.exists():
            print(f"Found: {loc}")
            if loc.is_dir():
                files = list(loc.rglob("*"))[:10]
                for f in files:
                    if f.is_file():
                        print(f"  - {f.relative_to(loc)} ({f.stat().st_size} bytes)")


def run_session_1():
    """First session: Teach the agent new information."""
    print("=" * 60)
    print("Long-term Memory: Session 1 (Teaching)")
    print("=" * 60)
    print("""
In this session, we'll teach the agent information that should
be remembered in the next session.
""")

    # Agent with long-term memory
    learner = Agent(
        role="Long-term Memory Assistant",
        goal="Learn and remember information across sessions",
        backstory="""You have an excellent long-term memory.
        Information you learn is remembered forever and can be
        recalled in future conversations.""",
        verbose=True,
        memory=True,
    )

    # Task to learn new information
    learning_task = Task(
        description="""Learn and commit to long-term memory:

        Company Profile - TechCorp Inc:
        - Founded: 2015
        - CEO: Michael Roberts
        - Headquarters: San Francisco
        - Revenue: $500 million
        - Employees: 2,500
        - Main Product: CloudSync Platform
        - Stock Symbol: TECH

        User Preferences:
        - Preferred name: Alex
        - Communication style: Formal
        - Time zone: Pacific (PST)
        - Interests: AI, Machine Learning, Startups

        Confirm that all information has been stored in long-term memory.""",
        expected_output="Confirmation that all information is memorized",
        agent=learner,
    )

    crew = Crew(
        agents=[learner],
        tasks=[learning_task],
        verbose=True,
        memory=True,
        # Long-term memory configuration
        # Note: The exact configuration may vary by CrewAI version
    )

    print("\nExecuting Session 1...")
    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("Session 1 Result:")
    print("=" * 60)
    print(result)

    check_memory_storage()

    print("\n" + "=" * 60)
    print("Session 1 Complete")
    print("=" * 60)
    print("Now run: python 12_memory_longterm.py --session 2")


def run_session_2():
    """Second session: Test recall from long-term memory."""
    print("=" * 60)
    print("Long-term Memory: Session 2 (Recall)")
    print("=" * 60)
    print("""
In this session, we'll test if the agent can recall information
learned in Session 1.
""")

    # Same agent configuration
    recaller = Agent(
        role="Long-term Memory Assistant",
        goal="Recall information from long-term memory",
        backstory="""You have an excellent long-term memory.
        You can recall information learned in previous sessions.""",
        verbose=True,
        memory=True,
    )

    # Task to recall information
    recall_task = Task(
        description="""From your long-term memory, please answer:

        1. What company did we discuss? What is their stock symbol?
        2. Who is the CEO?
        3. What is the user's preferred name?
        4. What are the user's interests?

        If you cannot recall something, say "Not in memory".
        This tests if long-term memory persists across sessions.""",
        expected_output="Answers to all questions from long-term memory",
        agent=recaller,
    )

    crew = Crew(
        agents=[recaller],
        tasks=[recall_task],
        verbose=True,
        memory=True,
    )

    print("\nExecuting Session 2...")
    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("Session 2 Result:")
    print("=" * 60)
    print(result)

    check_memory_storage()


def test_entity_memory():
    """Test entity-specific memory."""
    print("=" * 60)
    print("Entity Memory Test")
    print("=" * 60)
    print("""
Entity Memory tracks information about specific entities
(people, companies, concepts) mentioned in conversations.
""")

    assistant = Agent(
        role="Entity-Aware Assistant",
        goal="Track and remember information about entities",
        backstory="""You have excellent entity memory. You track
        details about people, companies, and concepts mentioned.""",
        verbose=True,
        memory=True,
    )

    # Task with multiple entity mentions
    entity_task = Task(
        description="""Process this conversation and track all entities:

        "I just met with John Smith from Acme Corp. He mentioned that
        their competitor, GlobalTech, is launching a new product called
        SmartWidget. John is interested in our AI solution and wants
        to set up a demo next week. He reports to Lisa Wang, the CTO."

        List all entities identified and what you learned about each.""",
        expected_output="A list of entities and their associated information",
        agent=assistant,
    )

    crew = Crew(
        agents=[assistant],
        tasks=[entity_task],
        verbose=True,
        memory=True,
    )

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("Entity Memory Result:")
    print("=" * 60)
    print(result)


def main():
    parser = argparse.ArgumentParser(description="Long-term Memory Test")
    parser.add_argument(
        "--session",
        type=int,
        choices=[1, 2],
        help="Session number (1=teach, 2=recall)",
    )
    parser.add_argument(
        "--entity",
        action="store_true",
        help="Run entity memory test",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check memory storage",
    )
    args = parser.parse_args()

    if args.check:
        check_memory_storage()
        return

    if args.entity:
        test_entity_memory()
        return

    if args.session == 1:
        run_session_1()
    elif args.session == 2:
        run_session_2()
    else:
        print("""
Long-term Memory Test
=====================

Usage:
  python 12_memory_longterm.py --session 1   # First: Teach information
  python 12_memory_longterm.py --session 2   # Second: Test recall
  python 12_memory_longterm.py --entity      # Test entity memory
  python 12_memory_longterm.py --check       # Check memory storage

LangGraph Comparison:
- CrewAI: memory=True enables automatic long-term memory
- LangGraph: Requires explicit Checkpointer configuration

Note: Long-term memory effectiveness depends on CrewAI's
internal implementation and may require specific configuration.
""")


if __name__ == "__main__":
    main()
