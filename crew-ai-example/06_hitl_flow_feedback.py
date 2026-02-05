"""
06_hitl_flow_feedback.py - Human Feedback in Flow

Purpose: Verify HITL functionality in CrewAI Flow
- Flow state transitions
- approve/reject branching
- Human feedback loop

LangGraph Comparison:
- LangGraph: Command(resume=) to resume workflow
- CrewAI Flow: Build workflow with @start, @listen, @router

Note: CrewAI Flow is a relatively new feature, API may change
"""

from crewai.flow.flow import Flow, listen, start, router
from pydantic import BaseModel


class ProposalState(BaseModel):
    """State for the proposal workflow."""

    topic: str = "AI Agent Framework Adoption"  # Set default value
    draft: str = ""
    feedback: str = ""
    status: str = "pending"  # pending, approved, rejected, revised
    revision_count: int = 0


class ProposalFlow(Flow[ProposalState]):
    """
    A flow that demonstrates human feedback integration.

    Flow structure:
    1. generate_proposal -> draft created
    2. review_proposal -> human reviews and provides feedback
    3. Based on feedback:
       - If approved -> finalize
       - If rejected -> revise (up to 3 times)
    """

    @start()
    def generate_proposal(self):
        """Generate initial proposal."""
        print(f"\n[Flow] Generating proposal for: {self.state.topic}")

        self.state.draft = f"""
Proposal: {self.state.topic}
============================

Executive Summary:
This proposal outlines a comprehensive approach to {self.state.topic}.

Objectives:
1. Primary goal related to {self.state.topic}
2. Secondary objectives
3. Success metrics

Timeline:
- Phase 1: Planning and Research
- Phase 2: Implementation
- Phase 3: Review and Iteration

Budget Estimate:
[To be determined based on scope]

Revision #{self.state.revision_count}
"""
        print(f"[Flow] Draft generated (revision {self.state.revision_count})")

    @listen(generate_proposal)
    def review_proposal(self):
        """
        Human review step.
        In production, this would integrate with a UI or notification system.
        """
        print("\n" + "=" * 60)
        print("HUMAN REVIEW REQUIRED")
        print("=" * 60)
        print("\nCurrent Draft:")
        print(self.state.draft)
        print("\n" + "-" * 60)

        # Simulate human feedback input
        print("\nOptions:")
        print("1. Type 'approve' to accept the proposal")
        print("2. Type 'reject' to reject and end the flow")
        print("3. Type any feedback to request revisions")

        feedback = input("\nYour feedback: ").strip()

        if feedback.lower() == "approve":
            self.state.status = "approved"
            self.state.feedback = "Approved by reviewer"
        elif feedback.lower() == "reject":
            self.state.status = "rejected"
            self.state.feedback = "Rejected by reviewer"
        else:
            self.state.feedback = feedback
            self.state.status = "needs_revision"

    @router(review_proposal)
    def route_after_review(self):
        """Route based on review outcome."""
        if self.state.status == "approved":
            return "finalize"
        elif self.state.status == "rejected":
            return "handle_rejection"
        else:
            return "revise"

    @listen("finalize")
    def finalize_proposal(self):
        """Finalize the approved proposal."""
        print("\n[Flow] Proposal approved! Finalizing...")
        self.state.status = "finalized"
        return f"""
FINAL PROPOSAL
==============
{self.state.draft}

Status: APPROVED
Reviewed and approved for implementation.
"""

    @listen("handle_rejection")
    def handle_rejection(self):
        """Handle rejected proposal."""
        print("\n[Flow] Proposal rejected.")
        return f"""
PROPOSAL REJECTED
=================
Topic: {self.state.topic}
Feedback: {self.state.feedback}

The proposal has been rejected and will not proceed.
"""

    @listen("revise")
    def revise_proposal(self):
        """Revise the proposal based on feedback."""
        self.state.revision_count += 1

        if self.state.revision_count >= 3:
            print("\n[Flow] Maximum revisions reached. Auto-approving.")
            self.state.status = "approved"
            return "finalize"

        print(f"\n[Flow] Revising proposal (attempt {self.state.revision_count})")
        print(f"[Flow] Incorporating feedback: {self.state.feedback}")

        # Update draft with feedback
        self.state.draft = f"""
Proposal: {self.state.topic}
============================
[REVISED based on feedback: {self.state.feedback}]

Executive Summary:
This revised proposal addresses the feedback provided.

Objectives (Updated):
1. Refined goal based on reviewer feedback
2. Additional considerations
3. Updated success metrics

Timeline (Adjusted):
- Phase 1: Extended planning
- Phase 2: Implementation with checkpoints
- Phase 3: Comprehensive review

Budget Estimate:
[Updated based on revised scope]

Revision #{self.state.revision_count}
"""
        # Go back to review
        return self.review_proposal()


def main():
    print("=" * 60)
    print("HITL: Flow-based Human Feedback Test")
    print("=" * 60)
    print("""
This example demonstrates human-in-the-loop using CrewAI Flow.
The flow will pause at review points and wait for human input.

Flow Steps:
1. Generate proposal draft
2. Human reviews and provides feedback
3. Route based on feedback (approve/reject/revise)
4. Repeat revision up to 3 times if needed

LangGraph Comparison:
- CrewAI Flow: Decorators (@start, @listen, @router) define the graph
- LangGraph: StateGraph.add_node() and add_edge() for explicit graph
""")

    # Initialize flow (state uses default values from ProposalState)
    flow = ProposalFlow()

    print("\n" + "=" * 60)
    print("Starting Flow Execution")
    print("=" * 60)

    result = flow.kickoff()

    print("\n" + "=" * 60)
    print("Flow Result:")
    print("=" * 60)
    print(result)

    print("\n" + "=" * 60)
    print("Final State:")
    print("=" * 60)
    print(f"Status: {flow.state.status}")
    print(f"Revisions: {flow.state.revision_count}")
    print(f"Last Feedback: {flow.state.feedback}")


if __name__ == "__main__":
    main()
