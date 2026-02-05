"""
10_collaboration_hierarchical.py - Hierarchical Process

Purpose: Verify CrewAI's hierarchical process
- Behavior of Process.hierarchical
- Automatic manager agent generation
- Dynamic task assignment

CrewAI-specific feature:
- LangGraph has no built-in hierarchical process
- LangGraph requires manual supervisor pattern implementation
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool


# =============================================================================
# Tools for the team
# =============================================================================
@tool("Market Research")
def research_market(topic: str) -> str:
    """
    Research market trends and data for a given topic.

    Args:
        topic: The market topic to research
    """
    return f"""
Market Research Report: {topic}
================================
Market Size: $50 billion (2024)
Growth Rate: 15% CAGR
Key Players:
- Company A (30% market share)
- Company B (25% market share)
- Company C (20% market share)

Trends:
1. Increasing adoption of AI
2. Shift to cloud-based solutions
3. Growing demand for automation
"""


@tool("Technical Analysis")
def analyze_technology(technology: str) -> str:
    """
    Analyze a technology for feasibility and implementation.

    Args:
        technology: The technology to analyze
    """
    return f"""
Technical Analysis: {technology}
=================================
Maturity Level: Growth stage
Implementation Complexity: Medium
Required Skills:
- Python/JavaScript development
- Cloud infrastructure
- API integration

Timeline Estimate:
- MVP: 3-4 months
- Full Product: 8-12 months

Risks:
1. Integration challenges
2. Scalability considerations
"""


@tool("Financial Model")
def create_financial_model(project: str) -> str:
    """
    Create a financial model for a project.

    Args:
        project: The project to model
    """
    return f"""
Financial Model: {project}
===========================
Initial Investment: $500,000
Monthly Operating Cost: $50,000

Revenue Projections:
- Year 1: $200,000
- Year 2: $800,000
- Year 3: $2,000,000

Break-even: Month 18
ROI (3 year): 180%

Assumptions:
- 20% monthly growth rate
- 30% gross margin
"""


def main():
    print("=" * 60)
    print("Role-Based Collaboration: Hierarchical Process")
    print("=" * 60)
    print("""
This example demonstrates CrewAI's hierarchical process where a
manager agent automatically coordinates task assignment to workers.

Key Points:
- Process.hierarchical creates a manager automatically
- Manager decides which agent handles each task
- Tasks are dynamically assigned based on agent capabilities

LangGraph Comparison:
- CrewAI: Process.hierarchical (automatic manager)
- LangGraph: Supervisor pattern (manual implementation)
""")

    # ==========================================================================
    # Define worker agents (no need to define manager - it's automatic)
    # ==========================================================================

    market_researcher = Agent(
        role="Market Research Analyst",
        goal="Provide comprehensive market analysis and competitive intelligence",
        backstory="""You are an experienced market research analyst with
        deep knowledge of technology markets. You excel at identifying
        trends, analyzing competition, and providing actionable insights.""",
        tools=[research_market],
        verbose=True,
    )

    tech_lead = Agent(
        role="Technical Lead",
        goal="Evaluate technical feasibility and provide implementation guidance",
        backstory="""You are a senior technical lead with experience in
        building software products. You can assess technical requirements,
        identify risks, and create implementation roadmaps.""",
        tools=[analyze_technology],
        verbose=True,
    )

    financial_analyst = Agent(
        role="Financial Analyst",
        goal="Create financial projections and analyze business viability",
        backstory="""You are a financial analyst specializing in tech
        startups. You create financial models, analyze ROI, and help
        make data-driven business decisions.""",
        tools=[create_financial_model],
        verbose=True,
    )

    # ==========================================================================
    # Define tasks (manager will assign these to appropriate agents)
    # ==========================================================================

    # Note: In hierarchical process, we don't assign agents to tasks
    # The manager will decide who handles what

    market_analysis_task = Task(
        description="""Analyze the AI agent framework market:
        - Current market size and growth rate
        - Key players and their market share
        - Major trends and opportunities""",
        expected_output="A detailed market analysis report",
        # No agent assignment - manager decides
    )

    technical_assessment_task = Task(
        description="""Assess the technical requirements for building
        an AI agent orchestration platform:
        - Technology stack recommendations
        - Implementation complexity
        - Required team skills""",
        expected_output="A technical assessment document",
        # No agent assignment
    )

    financial_projection_task = Task(
        description="""Create financial projections for an AI agent
        platform startup:
        - Initial investment requirements
        - Revenue projections for 3 years
        - Break-even analysis""",
        expected_output="A financial model with projections",
        # No agent assignment
    )

    final_report_task = Task(
        description="""Compile all analyses into a comprehensive
        business case document that includes:
        - Executive summary
        - Market opportunity
        - Technical approach
        - Financial viability
        - Recommendation""",
        expected_output="A complete business case document",
        # This may be assigned to any agent or handled by manager
    )

    # ==========================================================================
    # Create hierarchical crew
    # ==========================================================================
    crew = Crew(
        agents=[market_researcher, tech_lead, financial_analyst],
        tasks=[
            market_analysis_task,
            technical_assessment_task,
            financial_projection_task,
            final_report_task,
        ],
        process=Process.hierarchical,  # <-- Hierarchical process
        verbose=True,
        manager_llm="gpt-4o",  # Required for hierarchical process
    )

    print("\n" + "=" * 60)
    print("Executing Hierarchical Crew")
    print("=" * 60)
    print("""
Watch for manager behavior:
- Manager agent will be automatically created
- Manager will analyze tasks and assign to appropriate workers
- Manager will coordinate and compile final output
""")

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    print(result)

    # Show usage metrics if available
    if hasattr(result, "token_usage"):
        print("\n" + "=" * 60)
        print("Token Usage:")
        print("=" * 60)
        print(result.token_usage)


def compare_sequential():
    """Compare with sequential process for reference."""
    print("\n" + "=" * 60)
    print("Comparison: Sequential Process")
    print("=" * 60)

    # In sequential, each task must have an assigned agent
    researcher = Agent(
        role="Researcher",
        goal="Research and analyze",
        backstory="You are a researcher.",
        verbose=True,
    )

    task = Task(
        description="Briefly describe the AI agent market.",
        expected_output="A short summary",
        agent=researcher,  # Required for sequential
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,  # Sequential process
        verbose=True,
    )

    result = crew.kickoff()
    print("\nSequential Result:")
    print(result)


if __name__ == "__main__":
    main()
    # Uncomment to see sequential comparison:
    # compare_sequential()
