"""
LangGraph Governance - Policy Engine (GV-02, GV-03)
Least privilege, scoped permissions, and Policy as Code.

Evaluation: GV-02 (Least Privilege / Scope), GV-03 (Policy as Code)
"""

from typing import Annotated, TypedDict, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


# =============================================================================
# PERMISSION SYSTEM (GV-02)
# =============================================================================

class Permission(Enum):
    """Available permissions."""
    READ_FILE = "read:file"
    WRITE_FILE = "write:file"
    DELETE_FILE = "delete:file"
    READ_DATABASE = "read:database"
    WRITE_DATABASE = "write:database"
    DELETE_DATABASE = "delete:database"
    SEND_EMAIL = "send:email"
    SEND_BULK_EMAIL = "send:bulk_email"
    ADMIN = "admin:*"


@dataclass
class Scope:
    """Permission scope with resource restrictions."""
    permissions: Set[Permission]
    allowed_paths: Optional[Set[str]] = None  # e.g., {"/tmp/*", "/data/public/*"}
    allowed_tables: Optional[Set[str]] = None  # e.g., {"users", "orders"}
    max_recipients: int = 10  # For email
    purpose: Optional[str] = None  # Purpose binding


@dataclass
class Principal:
    """Identity with assigned scope."""
    id: str
    name: str
    scope: Scope
    created_at: datetime = field(default_factory=datetime.now)


class PermissionManager:
    """Manages permissions and scope checking."""

    def __init__(self):
        self.principals: dict[str, Principal] = {}

    def create_principal(self, id: str, name: str, scope: Scope) -> Principal:
        """Create a new principal with scope."""
        principal = Principal(id=id, name=name, scope=scope)
        self.principals[id] = principal
        return principal

    def check_permission(
        self,
        principal_id: str,
        required_permission: Permission,
        resource: Optional[str] = None,
        context: Optional[dict] = None
    ) -> tuple[bool, str]:
        """
        Check if principal has permission for resource.
        Returns: (allowed, reason)
        """
        principal = self.principals.get(principal_id)
        if not principal:
            return False, f"Unknown principal: {principal_id}"

        scope = principal.scope

        # Check admin permission
        if Permission.ADMIN in scope.permissions:
            return True, "Admin access"

        # Check specific permission
        if required_permission not in scope.permissions:
            return False, f"Permission denied: {required_permission.value} not in scope"

        # Check resource-specific restrictions
        if resource:
            # Path restriction
            if scope.allowed_paths and required_permission in [
                Permission.READ_FILE, Permission.WRITE_FILE, Permission.DELETE_FILE
            ]:
                if not self._path_matches(resource, scope.allowed_paths):
                    return False, f"Path not in allowed scope: {resource}"

            # Table restriction
            if scope.allowed_tables and required_permission in [
                Permission.READ_DATABASE, Permission.WRITE_DATABASE, Permission.DELETE_DATABASE
            ]:
                if resource not in scope.allowed_tables:
                    return False, f"Table not in allowed scope: {resource}"

        # Context-specific checks
        if context:
            if "recipients" in context and context["recipients"] > scope.max_recipients:
                return False, f"Exceeds max recipients: {context['recipients']} > {scope.max_recipients}"

        return True, "Permission granted"

    @staticmethod
    def _path_matches(path: str, patterns: Set[str]) -> bool:
        """Check if path matches any pattern."""
        for pattern in patterns:
            if pattern.endswith("/*"):
                prefix = pattern[:-2]
                if path.startswith(prefix):
                    return True
            elif path == pattern:
                return True
        return False


# =============================================================================
# POLICY AS CODE (GV-03)
# =============================================================================

@dataclass
class PolicyRule:
    """Single policy rule in Policy as Code."""
    id: str
    name: str
    description: str
    condition: str  # Python expression
    action: str  # "allow", "deny", "require_approval"
    priority: int = 0  # Higher priority rules evaluated first


class PolicyEngine:
    """
    Policy as Code engine.
    Implements GV-03: Declarative policy management.
    """

    def __init__(self):
        self.rules: list[PolicyRule] = []

    def add_rule(self, rule: PolicyRule):
        """Add a policy rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def load_policies_from_yaml(self, yaml_content: str):
        """Load policies from YAML (simplified for demo)."""
        # In production, use actual YAML parsing
        # This is a simplified demo
        pass

    def evaluate(
        self,
        tool_name: str,
        tool_args: dict,
        principal: Optional[Principal] = None,
        context: Optional[dict] = None
    ) -> tuple[str, str, Optional[PolicyRule]]:
        """
        Evaluate tool call against all policies.
        Returns: (action, reason, matched_rule)
        """
        eval_context = {
            "tool": tool_name,
            "args": tool_args,
            "principal": principal,
            "context": context or {},
            **tool_args  # Make args directly accessible
        }

        for rule in self.rules:
            try:
                # Evaluate condition
                if eval(rule.condition, {"__builtins__": {}}, eval_context):
                    return rule.action, f"Rule '{rule.name}': {rule.description}", rule
            except Exception as e:
                print(f"  [POLICY] Error evaluating rule {rule.id}: {e}")
                continue

        return "allow", "No matching policy (default allow)", None


# =============================================================================
# STATE AND TOOLS
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    principal_id: str
    policy_decisions: list


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return f"Contents of {path}: [file data]"


@tool
def write_file(path: str, content: str) -> str:
    """Write to a file."""
    return f"Written {len(content)} bytes to {path}"


@tool
def delete_file(path: str) -> str:
    """Delete a file."""
    return f"Deleted: {path}"


@tool
def query_database(table: str, query: str) -> str:
    """Query a database table."""
    return f"Query result from {table}: [10 rows]"


@tool
def update_database(table: str, data: str) -> str:
    """Update a database table."""
    return f"Updated {table} with data"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}: {subject}"


@tool
def send_bulk_email(recipients: int, subject: str) -> str:
    """Send bulk email to multiple recipients."""
    return f"Bulk email sent to {recipients} recipients: {subject}"


tools = [read_file, write_file, delete_file, query_database, update_database, send_email, send_bulk_email]


# =============================================================================
# INITIALIZE PERMISSION AND POLICY SYSTEMS
# =============================================================================

permission_manager = PermissionManager()
policy_engine = PolicyEngine()

# Create principals with different scopes
permission_manager.create_principal(
    "user_readonly",
    "Read-Only User",
    Scope(
        permissions={Permission.READ_FILE, Permission.READ_DATABASE},
        allowed_paths={"/data/public/*", "/tmp/*"},
        allowed_tables={"products", "categories"}
    )
)

permission_manager.create_principal(
    "user_standard",
    "Standard User",
    Scope(
        permissions={
            Permission.READ_FILE, Permission.WRITE_FILE,
            Permission.READ_DATABASE, Permission.WRITE_DATABASE,
            Permission.SEND_EMAIL
        },
        allowed_paths={"/data/user/*", "/tmp/*"},
        allowed_tables={"users", "orders"},
        max_recipients=10
    )
)

permission_manager.create_principal(
    "user_admin",
    "Admin User",
    Scope(permissions={Permission.ADMIN})
)

# Define policies (Policy as Code)
policy_engine.add_rule(PolicyRule(
    id="deny_root_access",
    name="Deny Root Access",
    description="Never allow access to system directories",
    condition="tool in ['read_file', 'write_file', 'delete_file'] and args.get('path', '').startswith('/etc')",
    action="deny",
    priority=100
))

policy_engine.add_rule(PolicyRule(
    id="require_approval_delete",
    name="Require Approval for Delete",
    description="All delete operations require approval",
    condition="tool in ['delete_file', 'update_database'] and 'delete' in str(args).lower()",
    action="require_approval",
    priority=50
))

policy_engine.add_rule(PolicyRule(
    id="limit_bulk_email",
    name="Limit Bulk Email",
    description="Bulk email over 50 recipients requires approval",
    condition="tool == 'send_bulk_email' and args.get('recipients', 0) > 50",
    action="require_approval",
    priority=40
))

policy_engine.add_rule(PolicyRule(
    id="allow_temp_files",
    name="Allow Temp Files",
    description="Always allow operations on /tmp",
    condition="tool in ['read_file', 'write_file'] and args.get('path', '').startswith('/tmp')",
    action="allow",
    priority=30
))


# =============================================================================
# GRAPH NODES
# =============================================================================

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


def agent(state: State) -> State:
    """Agent node."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def policy_check(state: State) -> Command:
    """
    Policy check node - combines permission and policy evaluation.
    Implements GV-02 (Least Privilege) and GV-03 (Policy as Code).
    """
    last_message = state["messages"][-1]
    principal_id = state.get("principal_id", "user_standard")
    decisions = state.get("policy_decisions", [])

    if not last_message.tool_calls:
        return Command(goto="tools")

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    print(f"\n  [POLICY CHECK] Tool: {tool_name}")
    print(f"  [POLICY CHECK] Principal: {principal_id}")
    print(f"  [POLICY CHECK] Args: {tool_args}")

    # Step 1: Check permissions (GV-02)
    permission_map = {
        "read_file": Permission.READ_FILE,
        "write_file": Permission.WRITE_FILE,
        "delete_file": Permission.DELETE_FILE,
        "query_database": Permission.READ_DATABASE,
        "update_database": Permission.WRITE_DATABASE,
        "send_email": Permission.SEND_EMAIL,
        "send_bulk_email": Permission.SEND_BULK_EMAIL,
    }

    required_perm = permission_map.get(tool_name)
    resource = tool_args.get("path") or tool_args.get("table")

    if required_perm:
        allowed, reason = permission_manager.check_permission(
            principal_id,
            required_perm,
            resource,
            tool_args
        )
        print(f"  [PERMISSION] {reason}")

        if not allowed:
            decisions.append({
                "tool": tool_name,
                "decision": "denied",
                "reason": reason,
                "type": "permission"
            })
            rejection_msg = ToolMessage(
                content=f"Permission denied: {reason}",
                tool_call_id=tool_call["id"]
            )
            return Command(
                goto="agent",
                update={"messages": [rejection_msg], "policy_decisions": decisions}
            )

    # Step 2: Check policies (GV-03)
    principal = permission_manager.principals.get(principal_id)
    action, reason, rule = policy_engine.evaluate(
        tool_name, tool_args, principal
    )
    print(f"  [POLICY] Action: {action}, Reason: {reason}")

    if action == "deny":
        decisions.append({
            "tool": tool_name,
            "decision": "denied",
            "reason": reason,
            "type": "policy",
            "rule_id": rule.id if rule else None
        })
        rejection_msg = ToolMessage(
            content=f"Policy violation: {reason}",
            tool_call_id=tool_call["id"]
        )
        return Command(
            goto="agent",
            update={"messages": [rejection_msg], "policy_decisions": decisions}
        )

    elif action == "require_approval":
        decision = interrupt({
            "action": "policy_approval",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "reason": reason,
            "rule_id": rule.id if rule else None
        })

        if decision.get("approved"):
            decisions.append({
                "tool": tool_name,
                "decision": "approved",
                "reason": reason,
                "type": "policy",
                "approver": decision.get("approver_id")
            })
            return Command(goto="tools", update={"policy_decisions": decisions})
        else:
            decisions.append({
                "tool": tool_name,
                "decision": "rejected",
                "reason": decision.get("rejection_reason", "Rejected by approver"),
                "type": "policy"
            })
            rejection_msg = ToolMessage(
                content=f"Request rejected: {decision.get('rejection_reason', 'No reason')}",
                tool_call_id=tool_call["id"]
            )
            return Command(
                goto="agent",
                update={"messages": [rejection_msg], "policy_decisions": decisions}
            )

    # Allowed
    decisions.append({
        "tool": tool_name,
        "decision": "allowed",
        "reason": reason,
        "type": "policy"
    })
    return Command(goto="tools", update={"policy_decisions": decisions})


def should_continue(state: State) -> str:
    """Conditional edge."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "policy_check"
    return END


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_graph():
    """Build graph with policy enforcement."""
    builder = StateGraph(State)

    builder.add_node("agent", agent)
    builder.add_node("policy_check", policy_check)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["policy_check", END])
    builder.add_edge("policy_check", "tools")
    builder.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# TESTS
# =============================================================================

def test_permission_check():
    """Test permission-based access control (GV-02)."""
    print("\n" + "=" * 70)
    print("TEST: Permission Check (GV-02 Least Privilege)")
    print("=" * 70)

    # Test read-only user
    print("\n--- Read-Only User ---")
    allowed, reason = permission_manager.check_permission(
        "user_readonly", Permission.READ_FILE, "/data/public/report.txt"
    )
    print(f"Read public file: {allowed} - {reason}")

    allowed, reason = permission_manager.check_permission(
        "user_readonly", Permission.WRITE_FILE, "/data/public/report.txt"
    )
    print(f"Write public file: {allowed} - {reason}")

    allowed, reason = permission_manager.check_permission(
        "user_readonly", Permission.READ_FILE, "/etc/passwd"
    )
    print(f"Read system file: {allowed} - {reason}")

    # Test standard user
    print("\n--- Standard User ---")
    allowed, reason = permission_manager.check_permission(
        "user_standard", Permission.WRITE_FILE, "/data/user/notes.txt"
    )
    print(f"Write user file: {allowed} - {reason}")

    allowed, reason = permission_manager.check_permission(
        "user_standard", Permission.WRITE_DATABASE, "users"
    )
    print(f"Write users table: {allowed} - {reason}")

    allowed, reason = permission_manager.check_permission(
        "user_standard", Permission.WRITE_DATABASE, "admin_logs"
    )
    print(f"Write admin_logs table: {allowed} - {reason}")


def test_policy_evaluation():
    """Test policy-as-code evaluation (GV-03)."""
    print("\n" + "=" * 70)
    print("TEST: Policy Evaluation (GV-03 Policy as Code)")
    print("=" * 70)

    test_cases = [
        ("read_file", {"path": "/etc/passwd"}, "Should deny - system file"),
        ("read_file", {"path": "/tmp/test.txt"}, "Should allow - temp file"),
        ("delete_file", {"path": "/data/file.txt"}, "Should require approval"),
        ("send_bulk_email", {"recipients": 100, "subject": "Test"}, "Should require approval"),
        ("send_bulk_email", {"recipients": 10, "subject": "Test"}, "Should allow"),
    ]

    for tool_name, args, description in test_cases:
        action, reason, rule = policy_engine.evaluate(tool_name, args)
        rule_id = rule.id if rule else "default"
        print(f"\n{description}")
        print(f"  Tool: {tool_name}, Args: {args}")
        print(f"  Result: {action} (Rule: {rule_id})")
        print(f"  Reason: {reason}")


def test_integrated_policy():
    """Test integrated permission + policy check."""
    print("\n" + "=" * 70)
    print("TEST: Integrated Policy Enforcement")
    print("=" * 70)

    graph = build_graph()

    # Test 1: Read-only user trying to write
    print("\n--- Test: Read-only user writing ---")
    config1 = {"configurable": {"thread_id": "policy-test-1"}}
    result = graph.invoke(
        {
            "messages": [("user", "Write 'hello' to /tmp/test.txt")],
            "principal_id": "user_readonly",
            "policy_decisions": []
        },
        config=config1
    )
    print(f"Decisions: {result.get('policy_decisions', [])}")

    # Test 2: Standard user accessing allowed path
    print("\n--- Test: Standard user on allowed path ---")
    config2 = {"configurable": {"thread_id": "policy-test-2"}}
    result = graph.invoke(
        {
            "messages": [("user", "Write 'hello' to /tmp/test.txt")],
            "principal_id": "user_standard",
            "policy_decisions": []
        },
        config=config2
    )
    print(f"Decisions: {result.get('policy_decisions', [])}")

    # Test 3: Admin user (should bypass most checks)
    print("\n--- Test: Admin user ---")
    config3 = {"configurable": {"thread_id": "policy-test-3"}}
    result = graph.invoke(
        {
            "messages": [("user", "Read /etc/passwd")],
            "principal_id": "user_admin",
            "policy_decisions": []
        },
        config=config3
    )
    print(f"Decisions: {result.get('policy_decisions', [])}")


def test_scope_inheritance():
    """Test scope restrictions."""
    print("\n" + "=" * 70)
    print("TEST: Scope Restrictions")
    print("=" * 70)

    # Create user with specific scope
    permission_manager.create_principal(
        "project_user",
        "Project User",
        Scope(
            permissions={Permission.READ_FILE, Permission.WRITE_FILE},
            allowed_paths={"/projects/alpha/*"},
            purpose="Project Alpha development"
        )
    )

    test_paths = [
        "/projects/alpha/src/main.py",
        "/projects/alpha/docs/readme.md",
        "/projects/beta/src/main.py",  # Different project
        "/home/user/personal.txt",  # Outside scope
    ]

    for path in test_paths:
        allowed, reason = permission_manager.check_permission(
            "project_user", Permission.WRITE_FILE, path
        )
        status = "✅" if allowed else "❌"
        print(f"{status} {path}: {reason}")


# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ GV-02 & GV-03: POLICY ENGINE - EVALUATION SUMMARY                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LangGraph Native Support: ⭐ (Not Supported)                                │
│                                                                             │
│ LangGraph does NOT provide:                                                 │
│   ❌ Permission system                                                      │
│   ❌ Scope/resource restrictions                                            │
│   ❌ Policy engine                                                          │
│   ❌ Policy as Code (YAML/DSL)                                              │
│   ❌ Principal/identity management                                          │
│                                                                             │
│ Custom Implementation Required:                                             │
│   ✅ Permission enum - Define available permissions                         │
│   ✅ Scope dataclass - Resource restrictions                                │
│   ✅ Principal - Identity with assigned scope                               │
│   ✅ PermissionManager - Check permissions against scope                    │
│   ✅ PolicyRule - Declarative policy rules                                  │
│   ✅ PolicyEngine - Evaluate conditions, return actions                     │
│   ✅ policy_check node - Integrate with interrupt() for approvals           │
│                                                                             │
│ GV-02 (Least Privilege) Features:                                           │
│   ✓ Permission-based access control                                         │
│   ✓ Resource-specific restrictions (paths, tables)                          │
│   ✓ Scope inheritance and limits                                            │
│   ✓ Purpose binding                                                         │
│                                                                             │
│ GV-03 (Policy as Code) Features:                                            │
│   ✓ Declarative policy rules                                                │
│   ✓ Condition expressions                                                   │
│   ✓ Priority-based evaluation                                               │
│   ✓ Actions: allow, deny, require_approval                                  │
│                                                                             │
│ Production Considerations:                                                  │
│   - External policy store (not hardcoded)                                   │
│   - YAML/Rego policy language                                               │
│   - Policy versioning and rollback                                          │
│   - Policy testing framework                                                │
│   - Integration with identity provider (OIDC, SAML)                         │
│   - Attribute-based access control (ABAC)                                   │
│                                                                             │
│ Rating:                                                                     │
│   GV-02 (Least Privilege): ⭐ (fully custom)                                │
│   GV-03 (Policy as Code): ⭐ (fully custom)                                 │
│                                                                             │
│   Works well once implemented, but requires significant effort.             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_permission_check()
    test_policy_evaluation()
    test_scope_inheritance()
    test_integrated_policy()

    print(SUMMARY)
