"""
14_governance_gate.py - Governance Gate for Destructive Operations (GV-01)

Purpose: Verify gate functionality for destructive operations
- human_input=True for approval before destructive operations
- Plan→Approval→Execute two-stage pattern
- Conditional approval (re-approval when threshold exceeded)
- Policy evaluation before tool execution

This is a FAIL-CLOSE requirement: destructive operations must require explicit approval.
"""

from typing import Type, Optional
from datetime import datetime
import json

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crewai.flow.flow import Flow, listen, start, router


# =============================================================================
# Tool Input Schemas
# =============================================================================

class FileDeleteInput(BaseModel):
    """Input schema for file deletion tool."""
    file_path: str = Field(..., description="Path to file to delete")
    force: bool = Field(default=False, description="Force delete without confirmation")


class BatchDeleteInput(BaseModel):
    """Input schema for batch deletion tool."""
    file_paths: list[str] = Field(..., description="List of file paths to delete")


class APICallInput(BaseModel):
    """Input schema for API call tool."""
    endpoint: str = Field(..., description="API endpoint to call")
    method: str = Field(default="GET", description="HTTP method")
    data: Optional[dict] = Field(default=None, description="Request body data")


# =============================================================================
# Dangerous Tools with Gate Mechanism
# =============================================================================

class DangerousFileDeleteTool(BaseTool):
    """
    A dangerous tool that simulates file deletion.
    Demonstrates gate mechanism for destructive operations.
    """

    name: str = "Dangerous File Delete"
    description: str = """Delete a file from the system. THIS IS A DESTRUCTIVE OPERATION.
    Use with caution. Args: file_path (required), force (optional, default False)"""
    args_schema: Type[BaseModel] = FileDeleteInput

    # Gate configuration
    requires_approval: bool = True
    approval_log: list = []

    def _run(self, file_path: str, force: bool = False) -> str:
        """Execute file deletion (simulated)."""
        operation = {
            "tool": self.name,
            "file_path": file_path,
            "force": force,
            "timestamp": datetime.now().isoformat(),
        }

        # Log the operation attempt
        self.approval_log.append(operation)
        print(f"  [DangerousFileDelete] Requested: {file_path} (force={force})")

        # Simulate the deletion (not actually deleting)
        return f"[SIMULATED] File '{file_path}' would be deleted. (force={force})"


class BatchDeleteTool(BaseTool):
    """
    Batch delete multiple files. Requires approval when count exceeds threshold.
    """

    name: str = "Batch File Delete"
    description: str = """Delete multiple files at once. THIS IS A DESTRUCTIVE OPERATION.
    Requires re-approval if deleting more than 5 files."""
    args_schema: Type[BaseModel] = BatchDeleteInput

    approval_threshold: int = 5

    def _run(self, file_paths: list[str]) -> str:
        """Execute batch deletion (simulated)."""
        count = len(file_paths)

        print(f"  [BatchDelete] Requested deletion of {count} files")

        if count > self.approval_threshold:
            print(f"  [BatchDelete] WARNING: Exceeds threshold ({self.approval_threshold})")
            print(f"  [BatchDelete] Additional approval would be required in production")

        return f"[SIMULATED] Would delete {count} files: {file_paths[:3]}{'...' if count > 3 else ''}"


class DangerousAPITool(BaseTool):
    """
    API call tool with policy evaluation.
    Demonstrates policy-based gating.
    """

    name: str = "API Caller"
    description: str = """Make API calls. POST/PUT/DELETE methods require approval."""
    args_schema: Type[BaseModel] = APICallInput

    # Policy: methods requiring approval
    dangerous_methods: list = ["POST", "PUT", "DELETE", "PATCH"]

    def _run(self, endpoint: str, method: str = "GET", data: Optional[dict] = None) -> str:
        """Execute API call (simulated)."""
        is_dangerous = method.upper() in self.dangerous_methods

        print(f"  [APICaller] {method} {endpoint}")
        if is_dangerous:
            print(f"  [APICaller] WARNING: {method} is a destructive method")

        return f"[SIMULATED] {method} {endpoint} - Data: {data}"


# =============================================================================
# Approval Flow State
# =============================================================================

class ApprovalFlowState(BaseModel):
    """State for the approval workflow."""

    operation_type: str = ""
    operation_details: dict = {}
    plan_description: str = ""
    is_approved: bool = False
    approval_timestamp: str = ""
    approver: str = ""
    rejection_reason: str = ""
    execution_result: str = ""


# =============================================================================
# Plan-Approval-Execute Flow
# =============================================================================

class ApprovalFlow(Flow[ApprovalFlowState]):
    """
    Two-stage execution flow: Plan → Approval → Execute

    This demonstrates the GV-01 requirement for destructive operation gates.
    """

    @start()
    def create_plan(self):
        """Step 1: Create execution plan."""
        print("\n[Plan Phase] Creating execution plan...")

        # Simulate plan creation
        self.state.plan_description = f"""
Execution Plan
==============
Operation: {self.state.operation_type}
Details: {json.dumps(self.state.operation_details, indent=2)}

Impact Assessment:
- This operation will modify system state
- Changes may not be reversible
- Requires human approval before execution

Recommended Action: Review and approve/reject
"""
        print(self.state.plan_description)

    @listen(create_plan)
    def request_approval(self):
        """Step 2: Request human approval."""
        print("\n" + "=" * 60)
        print("APPROVAL REQUIRED")
        print("=" * 60)
        print(f"\nOperation: {self.state.operation_type}")
        print(f"Details: {self.state.operation_details}")
        print("\nOptions:")
        print("  1. Type 'approve' to proceed")
        print("  2. Type 'reject' to cancel")
        print("  3. Type anything else for more details")

        response = input("\nYour decision: ").strip().lower()

        if response == "approve":
            self.state.is_approved = True
            self.state.approval_timestamp = datetime.now().isoformat()
            self.state.approver = "human_reviewer"
            print("\n[Approval] Operation APPROVED")
        elif response == "reject":
            self.state.is_approved = False
            self.state.rejection_reason = "Rejected by reviewer"
            print("\n[Approval] Operation REJECTED")
        else:
            # In production, would show more details and re-prompt
            self.state.is_approved = False
            self.state.rejection_reason = f"Additional info requested: {response}"
            print(f"\n[Approval] Requested info: {response}")
            print("[Approval] Operation PENDING (defaulting to reject for safety)")

    @router(request_approval)
    def route_after_approval(self):
        """Route based on approval decision."""
        if self.state.is_approved:
            return "execute"
        else:
            return "cancel"

    @listen("execute")
    def execute_operation(self):
        """Step 3a: Execute the approved operation."""
        print("\n[Execute Phase] Executing approved operation...")

        # Simulate execution
        self.state.execution_result = f"""
Execution Complete
==================
Operation: {self.state.operation_type}
Status: SUCCESS (simulated)
Approved by: {self.state.approver}
Approved at: {self.state.approval_timestamp}
"""
        return self.state.execution_result

    @listen("cancel")
    def cancel_operation(self):
        """Step 3b: Cancel the rejected operation."""
        print("\n[Cancel Phase] Operation cancelled.")

        self.state.execution_result = f"""
Operation Cancelled
===================
Operation: {self.state.operation_type}
Reason: {self.state.rejection_reason}
Status: NOT EXECUTED
"""
        return self.state.execution_result


# =============================================================================
# Policy Evaluator
# =============================================================================

class PolicyEvaluator:
    """
    Evaluates operations against defined policies.
    Determines if approval is required before execution.
    """

    def __init__(self):
        self.policies = {
            "file_delete": {
                "requires_approval": True,
                "max_batch_size": 5,
                "forbidden_paths": ["/etc", "/usr", "/bin", "/root"],
            },
            "api_call": {
                "requires_approval_methods": ["POST", "PUT", "DELETE", "PATCH"],
                "allowed_endpoints": ["*"],
                "rate_limit": 100,
            },
            "database": {
                "requires_approval": True,
                "read_only": False,
                "max_affected_rows": 1000,
            },
        }

    def evaluate(self, operation_type: str, details: dict) -> dict:
        """
        Evaluate an operation against policies.

        Returns:
            dict with 'allowed', 'requires_approval', and 'reason' keys
        """
        policy = self.policies.get(operation_type, {})

        result = {
            "allowed": True,
            "requires_approval": True,  # Default to requiring approval (fail-close)
            "reason": "",
            "warnings": [],
        }

        # Check file delete policies
        if operation_type == "file_delete":
            path = details.get("path", "")
            for forbidden in policy.get("forbidden_paths", []):
                if path.startswith(forbidden):
                    result["allowed"] = False
                    result["reason"] = f"Path '{forbidden}' is forbidden"
                    return result

            batch_size = details.get("batch_size", 1)
            if batch_size > policy.get("max_batch_size", 5):
                result["warnings"].append(
                    f"Batch size {batch_size} exceeds limit {policy['max_batch_size']}"
                )

        # Check API call policies
        if operation_type == "api_call":
            method = details.get("method", "GET").upper()
            if method in policy.get("requires_approval_methods", []):
                result["requires_approval"] = True
                result["reason"] = f"Method {method} requires approval"
            else:
                result["requires_approval"] = False

        return result


# =============================================================================
# Main Demonstrations
# =============================================================================

def demo_tool_with_gate():
    """Demonstrate destructive tools with human_input gate."""
    print("=" * 60)
    print("Demo 1: Destructive Tool with human_input=True")
    print("=" * 60)

    tools = [
        DangerousFileDeleteTool(),
        BatchDeleteTool(),
        DangerousAPITool(),
    ]

    # Agent that handles destructive operations
    operator = Agent(
        role="System Operator",
        goal="Execute system operations safely with proper approvals",
        backstory="""You are a careful system operator who always considers
        the impact of operations before executing them. You understand that
        destructive operations require approval.""",
        tools=tools,
        verbose=True,
    )

    # Task with human_input=True for approval gate
    task = Task(
        description="""Perform the following operations:
        1. Use the Dangerous File Delete tool to delete '/tmp/test.txt'
        2. Use the API Caller to make a DELETE request to '/api/users/123'

        Report what each tool would do (they are simulated).""",
        expected_output="Summary of simulated operations",
        agent=operator,
        human_input=True,  # GV-01: Require human approval
    )

    crew = Crew(
        agents=[operator],
        tasks=[task],
        verbose=True,
    )

    print("\n[Note] human_input=True will prompt for approval before task completion")
    print("[Note] This is the basic gate mechanism in CrewAI\n")

    result = crew.kickoff()
    print("\n" + "=" * 60)
    print("Result:")
    print(result)


def demo_approval_flow():
    """Demonstrate Plan-Approval-Execute flow pattern."""
    print("\n" + "=" * 60)
    print("Demo 2: Plan → Approval → Execute Flow")
    print("=" * 60)

    flow = ApprovalFlow()
    flow.state.operation_type = "file_delete"
    flow.state.operation_details = {
        "path": "/var/log/app.log",
        "reason": "Cleanup old logs",
    }

    result = flow.kickoff()
    print(result)


def demo_policy_evaluation():
    """Demonstrate policy-based gate evaluation."""
    print("\n" + "=" * 60)
    print("Demo 3: Policy-Based Gate Evaluation")
    print("=" * 60)

    evaluator = PolicyEvaluator()

    # Test cases
    test_cases = [
        ("file_delete", {"path": "/tmp/test.txt", "batch_size": 3}),
        ("file_delete", {"path": "/etc/passwd", "batch_size": 1}),
        ("file_delete", {"path": "/tmp/logs", "batch_size": 10}),
        ("api_call", {"method": "GET", "endpoint": "/api/users"}),
        ("api_call", {"method": "DELETE", "endpoint": "/api/users/123"}),
    ]

    for op_type, details in test_cases:
        result = evaluator.evaluate(op_type, details)
        print(f"\nOperation: {op_type}")
        print(f"  Details: {details}")
        print(f"  Allowed: {result['allowed']}")
        print(f"  Requires Approval: {result['requires_approval']}")
        if result['reason']:
            print(f"  Reason: {result['reason']}")
        if result['warnings']:
            print(f"  Warnings: {result['warnings']}")


def demo_conditional_approval():
    """Demonstrate conditional approval with threshold."""
    print("\n" + "=" * 60)
    print("Demo 4: Conditional Approval (Threshold-Based)")
    print("=" * 60)

    batch_tool = BatchDeleteTool()

    # Below threshold
    print("\n--- Batch size: 3 (below threshold of 5) ---")
    result1 = batch_tool._run(file_paths=["/tmp/a.txt", "/tmp/b.txt", "/tmp/c.txt"])
    print(f"Result: {result1}")

    # Above threshold
    print("\n--- Batch size: 8 (above threshold of 5) ---")
    result2 = batch_tool._run(file_paths=[f"/tmp/file{i}.txt" for i in range(8)])
    print(f"Result: {result2}")


def main():
    print("=" * 60)
    print("Governance Gate: Destructive Operation Verification (GV-01)")
    print("=" * 60)
    print("""
This script verifies the gate mechanism for destructive operations.

Verification Items:
1. human_input=True for destructive operation approval
2. Plan → Approval → Execute two-stage pattern
3. Conditional approval (re-approval when threshold exceeded)
4. Policy evaluation before tool execution

Key Points (Fail-Close):
- Destructive operations MUST require explicit approval
- Default behavior should be to deny/pause, not execute
- All operations should be logged for audit

LangGraph Comparison:
- LangGraph: interrupt() for pausing at any point
- CrewAI: human_input=True at task level, or Flow with @router
""")

    # Run demonstrations
    # Note: demo_tool_with_gate and demo_approval_flow require human interaction

    # Demo 3 and 4 don't require interaction
    demo_policy_evaluation()
    demo_conditional_approval()

    # Interactive demos (uncomment to run)
    print("\n" + "=" * 60)
    print("Interactive Demos (require human input)")
    print("=" * 60)
    print("\nTo run interactive demos, uncomment in main():")
    print("  - demo_tool_with_gate(): Tests human_input=True")
    print("  - demo_approval_flow(): Tests Plan→Approve→Execute flow")

    # Uncomment to run interactive demos:
    # demo_tool_with_gate()
    # demo_approval_flow()


if __name__ == "__main__":
    main()
