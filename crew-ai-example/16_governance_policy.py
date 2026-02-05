"""
16_governance_policy.py - Governance Policy (GV-02, GV-03)

Purpose: Verify least privilege and Policy as Code
- GV-02: Tool permission minimization
- GV-03: Declarative policy management (Policy as Code)
- Operation-level permission settings
- Prohibited/approval-required/threshold definitions

LangGraph Comparison:
- Both require custom implementation for policy enforcement
- CrewAI has no built-in policy engine
"""

import json
from pathlib import Path
from typing import Type, Optional, Any
from datetime import datetime
from enum import Enum

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# =============================================================================
# Policy Definitions
# =============================================================================

class PermissionLevel(str, Enum):
    """Permission levels for operations."""
    DENIED = "denied"           # Always blocked
    APPROVAL_REQUIRED = "approval_required"  # Needs human approval
    RESTRICTED = "restricted"   # Allowed with conditions
    ALLOWED = "allowed"         # Always allowed


class ToolPermission(BaseModel):
    """Permission configuration for a tool."""

    tool_name: str
    permission_level: PermissionLevel = PermissionLevel.ALLOWED
    allowed_operations: Optional[list[str]] = None  # Whitelist
    denied_operations: Optional[list[str]] = None   # Blacklist
    max_calls_per_minute: Optional[int] = None
    requires_audit: bool = False
    conditions: Optional[dict] = None  # Additional conditions


class PolicyConfig(BaseModel):
    """Complete policy configuration."""

    version: str = "1.0"
    name: str = "default"
    description: str = ""
    default_permission: PermissionLevel = PermissionLevel.RESTRICTED
    tool_permissions: list[ToolPermission] = []
    global_rules: dict = {}
    audit_all_operations: bool = True
    created_at: str = ""
    updated_at: str = ""


# =============================================================================
# Policy Engine
# =============================================================================

class PolicyEngine:
    """
    Policy as Code engine for tool permission management.

    Evaluates operations against declarative policies loaded from
    JSON/YAML configuration files.
    """

    def __init__(self, policy_path: Optional[Path] = None):
        self.policy: Optional[PolicyConfig] = None
        self.call_counts: dict[str, list[datetime]] = {}
        self.audit_log: list[dict] = []

        if policy_path:
            self.load_policy(policy_path)

    def load_policy(self, path: Path):
        """Load policy from JSON file."""
        if path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
                self.policy = PolicyConfig(**data)
        else:
            raise ValueError(f"Unsupported policy format: {path.suffix}")

        print(f"[PolicyEngine] Loaded policy: {self.policy.name} v{self.policy.version}")

    def load_policy_from_dict(self, data: dict):
        """Load policy from dictionary."""
        self.policy = PolicyConfig(**data)

    def get_tool_permission(self, tool_name: str) -> Optional[ToolPermission]:
        """Get permission config for a specific tool."""
        if not self.policy:
            return None

        for perm in self.policy.tool_permissions:
            if perm.tool_name == tool_name:
                return perm

        # Return default permission
        return ToolPermission(
            tool_name=tool_name,
            permission_level=self.policy.default_permission,
        )

    def check_rate_limit(self, tool_name: str, perm: ToolPermission) -> tuple[bool, str]:
        """Check if operation is within rate limits."""
        if perm.max_calls_per_minute is None:
            return True, ""

        now = datetime.now()
        if tool_name not in self.call_counts:
            self.call_counts[tool_name] = []

        # Remove calls older than 1 minute
        self.call_counts[tool_name] = [
            t for t in self.call_counts[tool_name]
            if (now - t).seconds < 60
        ]

        if len(self.call_counts[tool_name]) >= perm.max_calls_per_minute:
            return False, f"Rate limit exceeded: {perm.max_calls_per_minute}/min"

        self.call_counts[tool_name].append(now)
        return True, ""

    def evaluate(
        self,
        tool_name: str,
        operation: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """
        Evaluate an operation against the policy.

        Returns:
            dict with 'allowed', 'reason', 'requires_approval' keys
        """
        result = {
            "allowed": False,
            "reason": "",
            "requires_approval": False,
            "requires_audit": True,
            "warnings": [],
        }

        if not self.policy:
            result["allowed"] = True
            result["reason"] = "No policy loaded"
            return result

        perm = self.get_tool_permission(tool_name)
        if not perm:
            result["reason"] = "No permission configuration found"
            return result

        # Check permission level
        if perm.permission_level == PermissionLevel.DENIED:
            result["reason"] = f"Tool '{tool_name}' is denied by policy"
            return result

        if perm.permission_level == PermissionLevel.APPROVAL_REQUIRED:
            result["requires_approval"] = True
            result["allowed"] = True  # Allowed with approval
            result["reason"] = "Requires human approval"
            return result

        # Check operation whitelist/blacklist
        if operation:
            if perm.denied_operations and operation in perm.denied_operations:
                result["reason"] = f"Operation '{operation}' is denied"
                return result

            if perm.allowed_operations and operation not in perm.allowed_operations:
                result["reason"] = f"Operation '{operation}' not in allowed list"
                return result

        # Check rate limits
        rate_ok, rate_msg = self.check_rate_limit(tool_name, perm)
        if not rate_ok:
            result["reason"] = rate_msg
            return result

        # Check additional conditions
        if perm.conditions:
            for condition_key, condition_value in perm.conditions.items():
                if params and condition_key in params:
                    param_value = params[condition_key]
                    if isinstance(condition_value, dict):
                        if "max" in condition_value and param_value > condition_value["max"]:
                            result["reason"] = f"{condition_key} exceeds max ({condition_value['max']})"
                            return result
                        if "min" in condition_value and param_value < condition_value["min"]:
                            result["reason"] = f"{condition_key} below min ({condition_value['min']})"
                            return result

        result["allowed"] = True
        result["requires_audit"] = perm.requires_audit or self.policy.audit_all_operations
        return result

    def log_operation(self, tool_name: str, operation: str, params: dict,
                      result: dict, outcome: str):
        """Log operation for audit."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "operation": operation,
            "params": params,
            "policy_result": result,
            "outcome": outcome,
        }
        self.audit_log.append(entry)

    def get_audit_log(self) -> list[dict]:
        """Get all audit log entries."""
        return self.audit_log


# =============================================================================
# Policy-Aware Tool Wrapper
# =============================================================================

class PolicyAwareToolInput(BaseModel):
    """Input for policy-aware tool."""

    operation: str = Field(..., description="Operation to perform")
    data: Optional[dict] = Field(default=None, description="Operation data")


class PolicyAwareTool(BaseTool):
    """
    A tool that enforces policy checks before execution.

    Demonstrates least privilege principle (GV-02).
    """

    name: str = "Policy Aware Tool"
    description: str = """A tool that respects policy restrictions.
    Operations are checked against the policy engine before execution."""
    args_schema: Type[BaseModel] = PolicyAwareToolInput

    policy_engine: PolicyEngine = None

    def __init__(self, policy_engine: PolicyEngine, **kwargs):
        super().__init__(**kwargs)
        self.policy_engine = policy_engine

    def _run(self, operation: str, data: Optional[dict] = None) -> str:
        """Execute operation with policy check."""
        print(f"  [PolicyAwareTool] Checking policy for: {operation}")

        # Evaluate against policy
        result = self.policy_engine.evaluate(
            tool_name=self.name,
            operation=operation,
            params=data or {},
        )

        if not result["allowed"]:
            msg = f"[DENIED] {operation}: {result['reason']}"
            self.policy_engine.log_operation(
                self.name, operation, data or {}, result, "denied"
            )
            print(f"  [PolicyAwareTool] {msg}")
            return msg

        if result["requires_approval"]:
            msg = f"[APPROVAL REQUIRED] {operation}: Waiting for approval..."
            # In production, this would pause and wait for approval
            print(f"  [PolicyAwareTool] {msg}")
            # Simulating approval
            print(f"  [PolicyAwareTool] (Auto-approved for demo)")

        # Execute operation (simulated)
        output = f"[EXECUTED] {operation} with data: {data}"
        self.policy_engine.log_operation(
            self.name, operation, data or {}, result, "executed"
        )
        print(f"  [PolicyAwareTool] {output}")
        return output


# =============================================================================
# Restricted Tool Implementation
# =============================================================================

class RestrictedOperationInput(BaseModel):
    """Input for restricted operations."""

    action: str = Field(..., description="Action to perform: read, write, delete")
    resource: str = Field(..., description="Resource identifier")
    amount: Optional[int] = Field(default=None, description="Amount (for write operations)")


class RestrictedTool(BaseTool):
    """
    Tool with operation-level restrictions.

    - read: always allowed
    - write: restricted by amount
    - delete: requires approval
    """

    name: str = "Restricted Resource Tool"
    description: str = """Access resources with operation-level restrictions.
    read=allowed, write=restricted by amount, delete=requires approval"""
    args_schema: Type[BaseModel] = RestrictedOperationInput

    policy_engine: PolicyEngine = None

    def __init__(self, policy_engine: PolicyEngine, **kwargs):
        super().__init__(**kwargs)
        self.policy_engine = policy_engine

    def _run(self, action: str, resource: str, amount: Optional[int] = None) -> str:
        """Execute restricted operation."""
        print(f"  [RestrictedTool] {action} on {resource}")

        params = {"action": action, "resource": resource}
        if amount is not None:
            params["amount"] = amount

        result = self.policy_engine.evaluate(
            tool_name=self.name,
            operation=action,
            params=params,
        )

        if not result["allowed"]:
            return f"[BLOCKED] {action} on {resource}: {result['reason']}"

        if result["requires_approval"]:
            return f"[PENDING APPROVAL] {action} on {resource}"

        return f"[SUCCESS] {action} on {resource} (amount={amount})"


# =============================================================================
# Sample Policy Configuration
# =============================================================================

SAMPLE_POLICY = {
    "version": "1.0",
    "name": "production-policy",
    "description": "Production environment policy with strict controls",
    "default_permission": "restricted",
    "audit_all_operations": True,
    "tool_permissions": [
        {
            "tool_name": "Policy Aware Tool",
            "permission_level": "allowed",
            "allowed_operations": ["read", "list", "search"],
            "denied_operations": ["delete", "purge"],
            "max_calls_per_minute": 60,
            "requires_audit": True,
        },
        {
            "tool_name": "Restricted Resource Tool",
            "permission_level": "restricted",
            "allowed_operations": ["read", "write"],
            "denied_operations": ["delete"],
            "max_calls_per_minute": 30,
            "requires_audit": True,
            "conditions": {
                "amount": {"max": 1000, "min": 0},
            },
        },
        {
            "tool_name": "Dangerous Tool",
            "permission_level": "denied",
        },
        {
            "tool_name": "Admin Tool",
            "permission_level": "approval_required",
            "requires_audit": True,
        },
    ],
    "global_rules": {
        "max_concurrent_operations": 5,
        "require_mfa_for_destructive": True,
    },
    "created_at": datetime.now().isoformat(),
    "updated_at": datetime.now().isoformat(),
}


# =============================================================================
# Demonstrations
# =============================================================================

def demo_policy_loading():
    """Demonstrate policy loading and inspection."""
    print("=" * 60)
    print("Demo 1: Policy Loading (Policy as Code)")
    print("=" * 60)

    engine = PolicyEngine()
    engine.load_policy_from_dict(SAMPLE_POLICY)

    print(f"\nPolicy: {engine.policy.name} v{engine.policy.version}")
    print(f"Description: {engine.policy.description}")
    print(f"Default Permission: {engine.policy.default_permission}")
    print(f"Audit All: {engine.policy.audit_all_operations}")

    print("\nTool Permissions:")
    for perm in engine.policy.tool_permissions:
        print(f"  - {perm.tool_name}: {perm.permission_level.value}")
        if perm.allowed_operations:
            print(f"    Allowed: {perm.allowed_operations}")
        if perm.denied_operations:
            print(f"    Denied: {perm.denied_operations}")
        if perm.conditions:
            print(f"    Conditions: {perm.conditions}")


def demo_policy_evaluation():
    """Demonstrate policy evaluation for various operations."""
    print("\n" + "=" * 60)
    print("Demo 2: Policy Evaluation")
    print("=" * 60)

    engine = PolicyEngine()
    engine.load_policy_from_dict(SAMPLE_POLICY)

    test_cases = [
        ("Policy Aware Tool", "read", {}),
        ("Policy Aware Tool", "delete", {}),
        ("Restricted Resource Tool", "write", {"amount": 500}),
        ("Restricted Resource Tool", "write", {"amount": 2000}),  # Exceeds max
        ("Restricted Resource Tool", "delete", {}),
        ("Dangerous Tool", "anything", {}),
        ("Admin Tool", "configure", {}),
        ("Unknown Tool", "test", {}),
    ]

    for tool_name, operation, params in test_cases:
        result = engine.evaluate(tool_name, operation, params)
        status = "ALLOWED" if result["allowed"] else "DENIED"
        if result["requires_approval"]:
            status = "APPROVAL_REQUIRED"
        print(f"\n{tool_name}.{operation}({params}):")
        print(f"  Status: {status}")
        if result["reason"]:
            print(f"  Reason: {result['reason']}")


def demo_rate_limiting():
    """Demonstrate policy-based rate limiting."""
    print("\n" + "=" * 60)
    print("Demo 3: Policy Rate Limiting")
    print("=" * 60)

    # Create policy with low rate limit for demo
    policy = SAMPLE_POLICY.copy()
    policy["tool_permissions"] = [
        {
            "tool_name": "Rate Limited Tool",
            "permission_level": "allowed",
            "max_calls_per_minute": 3,  # Only 3 calls per minute
        }
    ]

    engine = PolicyEngine()
    engine.load_policy_from_dict(policy)

    print("\nAttempting 5 calls with limit of 3/minute:")
    for i in range(5):
        result = engine.evaluate("Rate Limited Tool", f"operation_{i}")
        status = "ALLOWED" if result["allowed"] else f"BLOCKED: {result['reason']}"
        print(f"  Call {i + 1}: {status}")


def demo_policy_aware_tools():
    """Demonstrate policy-aware tools in action."""
    print("\n" + "=" * 60)
    print("Demo 4: Policy-Aware Tool Execution")
    print("=" * 60)

    engine = PolicyEngine()
    engine.load_policy_from_dict(SAMPLE_POLICY)

    tool = PolicyAwareTool(policy_engine=engine)

    operations = [
        ("read", {"resource": "users"}),
        ("list", {"limit": 10}),
        ("delete", {"resource": "user_123"}),  # Should be denied
    ]

    for op, data in operations:
        print(f"\n--- Executing: {op} ---")
        result = tool._run(operation=op, data=data)
        print(f"Result: {result}")

    # Show audit log
    print("\n--- Audit Log ---")
    for entry in engine.get_audit_log():
        print(f"  {entry['timestamp']}: {entry['operation']} -> {entry['outcome']}")


def demo_save_policy():
    """Demonstrate saving policy to file."""
    print("\n" + "=" * 60)
    print("Demo 5: Save Policy to File (Policy as Code)")
    print("=" * 60)

    policy_dir = Path("./db/policies")
    policy_dir.mkdir(parents=True, exist_ok=True)

    policy_path = policy_dir / "sample_policy.json"

    with open(policy_path, "w") as f:
        json.dump(SAMPLE_POLICY, f, indent=2, default=str)

    print(f"Policy saved to: {policy_path}")
    print("\nPolicy content:")
    print(json.dumps(SAMPLE_POLICY, indent=2, default=str)[:500] + "...")


def main():
    print("=" * 60)
    print("Governance Policy Verification (GV-02, GV-03)")
    print("=" * 60)
    print("""
This script verifies least privilege and Policy as Code.

Verification Items:
- GV-02: Least Privilege
  - Tool permission minimization
  - Operation-level restrictions
  - Rate limiting per tool

- GV-03: Policy as Code
  - Declarative policy configuration
  - JSON/YAML policy files
  - Prohibited/approval-required/threshold definitions

Key Concepts:
- PolicyEngine: Central policy evaluation
- ToolPermission: Per-tool configuration
- PermissionLevel: denied/approval_required/restricted/allowed
- Audit logging: All operations logged

LangGraph Comparison:
- Neither has built-in policy engine
- Custom implementation required for both
- CrewAI's BaseTool can be extended for policy checks
""")

    # Run all demos
    demo_policy_loading()
    demo_policy_evaluation()
    demo_rate_limiting()
    demo_policy_aware_tools()
    demo_save_policy()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
