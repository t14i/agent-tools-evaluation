"""
LangGraph Connectors - Authentication & Credential Management (CX-01)
OAuth, token rotation, and secret management patterns.

Evaluation: CX-01 (Auth / Credential Management)
"""

import hashlib
import json
import time
from typing import Annotated, TypedDict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


# =============================================================================
# SECRET MANAGEMENT
# =============================================================================

class SecretType(Enum):
    """Types of secrets."""
    API_KEY = "api_key"
    OAUTH_TOKEN = "oauth_token"
    DATABASE_PASSWORD = "database_password"
    ENCRYPTION_KEY = "encryption_key"


@dataclass
class Secret:
    """A managed secret."""
    name: str
    type: SecretType
    value: str  # In production, this would be encrypted
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)
    version: int = 1


class SecretManager:
    """
    Manages secrets and credentials.
    In production, use AWS Secrets Manager, HashiCorp Vault, etc.
    """

    def __init__(self):
        self.secrets: dict[str, Secret] = {}
        self.lock = Lock()

    def store(
        self,
        name: str,
        value: str,
        type: SecretType,
        expires_in: Optional[timedelta] = None,
        metadata: dict = None
    ) -> Secret:
        """Store a secret."""
        with self.lock:
            expires_at = datetime.now() + expires_in if expires_in else None

            # Check if updating existing secret
            version = 1
            if name in self.secrets:
                version = self.secrets[name].version + 1

            secret = Secret(
                name=name,
                type=type,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                metadata=metadata or {},
                version=version
            )
            self.secrets[name] = secret
            print(f"  [SECRET] Stored: {name} (v{version})")
            return secret

    def get(self, name: str) -> Optional[str]:
        """Get a secret value."""
        with self.lock:
            secret = self.secrets.get(name)
            if not secret:
                return None

            # Check expiration
            if secret.expires_at and datetime.now() > secret.expires_at:
                print(f"  [SECRET] Expired: {name}")
                return None

            return secret.value

    def rotate(self, name: str, new_value: str) -> Optional[Secret]:
        """Rotate a secret to a new value."""
        with self.lock:
            if name not in self.secrets:
                return None

            old_secret = self.secrets[name]
            return self.store(
                name=name,
                value=new_value,
                type=old_secret.type,
                expires_in=timedelta(hours=1) if old_secret.expires_at else None,
                metadata=old_secret.metadata
            )

    def delete(self, name: str) -> bool:
        """Delete a secret."""
        with self.lock:
            if name in self.secrets:
                del self.secrets[name]
                print(f"  [SECRET] Deleted: {name}")
                return True
            return False


# =============================================================================
# OAUTH CLIENT
# =============================================================================

@dataclass
class OAuthToken:
    """OAuth token data."""
    access_token: str
    token_type: str
    expires_at: datetime
    refresh_token: Optional[str] = None
    scope: Optional[str] = None


class OAuthClient:
    """
    OAuth 2.0 client implementation.
    Handles token acquisition, refresh, and management.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        secret_manager: SecretManager
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.secret_manager = secret_manager
        self.lock = Lock()

    def get_token(self, service_name: str) -> Optional[str]:
        """Get a valid access token, refreshing if needed."""
        with self.lock:
            # Check if we have a valid token
            token = self.secret_manager.get(f"oauth_{service_name}_access")
            if token:
                return token

            # Try to refresh
            refresh_token = self.secret_manager.get(f"oauth_{service_name}_refresh")
            if refresh_token:
                print(f"  [OAUTH] Refreshing token for {service_name}")
                return self._refresh_token(service_name, refresh_token)

            print(f"  [OAUTH] No valid token for {service_name}")
            return None

    def authenticate(self, service_name: str, code: str) -> Optional[OAuthToken]:
        """Exchange authorization code for tokens."""
        print(f"  [OAUTH] Authenticating {service_name}")

        # Simulated token exchange
        token = OAuthToken(
            access_token=f"access_{hashlib.sha256(code.encode()).hexdigest()[:16]}",
            token_type="Bearer",
            expires_at=datetime.now() + timedelta(hours=1),
            refresh_token=f"refresh_{hashlib.sha256(code.encode()).hexdigest()[:16]}",
            scope="read write"
        )

        # Store tokens
        self._store_tokens(service_name, token)
        return token

    def _refresh_token(self, service_name: str, refresh_token: str) -> Optional[str]:
        """Refresh an access token."""
        # Simulated token refresh
        new_access_token = f"access_{hashlib.sha256(refresh_token.encode()).hexdigest()[:16]}"

        self.secret_manager.store(
            f"oauth_{service_name}_access",
            new_access_token,
            SecretType.OAUTH_TOKEN,
            expires_in=timedelta(hours=1)
        )

        return new_access_token

    def _store_tokens(self, service_name: str, token: OAuthToken):
        """Store OAuth tokens in secret manager."""
        self.secret_manager.store(
            f"oauth_{service_name}_access",
            token.access_token,
            SecretType.OAUTH_TOKEN,
            expires_in=timedelta(hours=1)
        )

        if token.refresh_token:
            self.secret_manager.store(
                f"oauth_{service_name}_refresh",
                token.refresh_token,
                SecretType.OAUTH_TOKEN,
                expires_in=timedelta(days=30)
            )

    def revoke(self, service_name: str):
        """Revoke tokens for a service."""
        self.secret_manager.delete(f"oauth_{service_name}_access")
        self.secret_manager.delete(f"oauth_{service_name}_refresh")
        print(f"  [OAUTH] Revoked tokens for {service_name}")


# =============================================================================
# CREDENTIAL PROVIDER
# =============================================================================

class CredentialProvider:
    """
    Provides credentials to tools at runtime.
    Implements just-in-time credential injection.
    """

    def __init__(self, secret_manager: SecretManager, oauth_client: OAuthClient):
        self.secret_manager = secret_manager
        self.oauth_client = oauth_client
        self.credential_map: dict[str, str] = {}  # tool_name -> secret_name

    def register_tool(self, tool_name: str, secret_name: str, auth_type: str = "api_key"):
        """Register a tool with its credential source."""
        self.credential_map[tool_name] = {
            "secret_name": secret_name,
            "auth_type": auth_type
        }

    def get_credentials(self, tool_name: str) -> Optional[dict]:
        """Get credentials for a tool."""
        if tool_name not in self.credential_map:
            return None

        config = self.credential_map[tool_name]
        auth_type = config["auth_type"]
        secret_name = config["secret_name"]

        if auth_type == "api_key":
            api_key = self.secret_manager.get(secret_name)
            return {"api_key": api_key} if api_key else None

        elif auth_type == "oauth":
            token = self.oauth_client.get_token(secret_name)
            return {"access_token": token} if token else None

        return None


# =============================================================================
# STATE AND TOOLS
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    authenticated_services: list


# Initialize systems
secret_manager = SecretManager()
oauth_client = OAuthClient(
    client_id="app_client_id",
    client_secret="app_client_secret",
    token_url="https://auth.example.com/oauth/token",
    secret_manager=secret_manager
)
credential_provider = CredentialProvider(secret_manager, oauth_client)

# Pre-configure some API keys
secret_manager.store("github_api_key", "ghp_xxxxxxxxxxxxxxxxxxxx", SecretType.API_KEY)
secret_manager.store("openai_api_key", "sk-xxxxxxxxxxxxxxxxxxxx", SecretType.API_KEY)

# Register tools with credentials
credential_provider.register_tool("github_api", "github_api_key", "api_key")
credential_provider.register_tool("slack_api", "slack", "oauth")


@tool
def github_api(query: str) -> str:
    """Query GitHub API (requires authentication)."""
    creds = credential_provider.get_credentials("github_api")
    if not creds:
        return "Error: GitHub API key not configured"
    # In production, use the API key for authentication
    return f"GitHub API result for '{query}' (authenticated with API key ...{creds['api_key'][-4:]})"


@tool
def slack_api(channel: str, message: str) -> str:
    """Send message to Slack (requires OAuth)."""
    creds = credential_provider.get_credentials("slack_api")
    if not creds:
        return "Error: Slack not authenticated. Please authorize first."
    return f"Message sent to #{channel}: {message} (using OAuth token)"


@tool
def authenticate_service(service: str, code: str) -> str:
    """Authenticate with an OAuth service."""
    token = oauth_client.authenticate(service, code)
    if token:
        return f"Successfully authenticated with {service}. Token expires at {token.expires_at}"
    return f"Failed to authenticate with {service}"


@tool
def rotate_api_key(service: str, new_key: str) -> str:
    """Rotate an API key."""
    secret_name = f"{service}_api_key"
    result = secret_manager.rotate(secret_name, new_key)
    if result:
        return f"API key for {service} rotated to version {result.version}"
    return f"Failed to rotate API key for {service}"


tools = [github_api, slack_api, authenticate_service, rotate_api_key]


# =============================================================================
# AUTHENTICATED TOOL NODE
# =============================================================================

class AuthenticatedToolNode:
    """Tool node that injects credentials before execution."""

    def __init__(
        self,
        tools: list,
        credential_provider: CredentialProvider
    ):
        self.tool_node = ToolNode(tools)
        self.tools_by_name = {t.name: t for t in tools}
        self.credential_provider = credential_provider

    def __call__(self, state: dict) -> dict:
        """Execute tools with credential injection."""
        last_message = state["messages"][-1]
        results = []
        authenticated = state.get("authenticated_services", [])

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # Check credentials
            creds = self.credential_provider.get_credentials(tool_name)
            if creds:
                print(f"  [AUTH] Credentials available for {tool_name}")
            else:
                print(f"  [AUTH] No credentials for {tool_name}")

            # Execute tool
            try:
                tool_fn = self.tools_by_name[tool_name]
                result = tool_fn.invoke(tool_args)
                results.append(ToolMessage(content=str(result), tool_call_id=tool_id))

                # Track authentication
                if creds and tool_name not in authenticated:
                    authenticated.append(tool_name)

            except Exception as e:
                results.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_id))

        return {"messages": results, "authenticated_services": authenticated}


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


def agent(state: State) -> State:
    """Agent node."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: State) -> str:
    """Conditional edge."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_graph():
    """Build graph with authenticated tools."""
    builder = StateGraph(State)

    builder.add_node("agent", agent)
    builder.add_node("tools", AuthenticatedToolNode(tools, credential_provider))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")

    return builder.compile()


# =============================================================================
# TESTS
# =============================================================================

def test_secret_management():
    """Test secret storage and retrieval."""
    print("\n" + "=" * 70)
    print("TEST: Secret Management")
    print("=" * 70)

    # Store a secret
    secret_manager.store(
        "test_api_key",
        "sk-test-12345",
        SecretType.API_KEY,
        expires_in=timedelta(hours=1)
    )

    # Retrieve
    value = secret_manager.get("test_api_key")
    print(f"\nRetrieved secret: {value[:8]}...")

    # Rotate
    secret_manager.rotate("test_api_key", "sk-test-67890")
    new_value = secret_manager.get("test_api_key")
    print(f"After rotation: {new_value[:8]}...")

    # Test expiration
    secret_manager.store(
        "expiring_secret",
        "temp-value",
        SecretType.API_KEY,
        expires_in=timedelta(seconds=-1)  # Already expired
    )
    expired = secret_manager.get("expiring_secret")
    print(f"Expired secret: {expired}")


def test_oauth_flow():
    """Test OAuth authentication flow."""
    print("\n" + "=" * 70)
    print("TEST: OAuth Flow")
    print("=" * 70)

    # Simulate authorization code
    auth_code = "authorization_code_from_callback"

    # Authenticate
    token = oauth_client.authenticate("google_drive", auth_code)
    print(f"\nAccess token: {token.access_token[:20]}...")
    print(f"Refresh token: {token.refresh_token[:20]}...")
    print(f"Expires at: {token.expires_at}")

    # Get token (should return cached)
    cached_token = oauth_client.get_token("google_drive")
    print(f"\nCached token: {cached_token[:20]}...")

    # Revoke
    oauth_client.revoke("google_drive")
    after_revoke = oauth_client.get_token("google_drive")
    print(f"After revoke: {after_revoke}")


def test_credential_provider():
    """Test credential provider."""
    print("\n" + "=" * 70)
    print("TEST: Credential Provider")
    print("=" * 70)

    # Get API key credentials
    github_creds = credential_provider.get_credentials("github_api")
    print(f"\nGitHub credentials: {github_creds}")

    # Get OAuth credentials (need to authenticate first)
    oauth_client.authenticate("slack", "slack_auth_code")
    slack_creds = credential_provider.get_credentials("slack_api")
    print(f"Slack credentials: {slack_creds}")

    # Unregistered tool
    unknown_creds = credential_provider.get_credentials("unknown_tool")
    print(f"Unknown tool credentials: {unknown_creds}")


def test_authenticated_execution():
    """Test tool execution with authentication."""
    print("\n" + "=" * 70)
    print("TEST: Authenticated Tool Execution")
    print("=" * 70)

    graph = build_graph()

    result = graph.invoke({
        "messages": [("user", "Search GitHub for 'langgraph'")],
        "authenticated_services": []
    })

    print(f"\nResult: {result['messages'][-1].content}")
    print(f"Authenticated services: {result.get('authenticated_services', [])}")


def test_token_rotation():
    """Test API key rotation."""
    print("\n" + "=" * 70)
    print("TEST: Token Rotation")
    print("=" * 70)

    # Initial key
    initial = secret_manager.get("github_api_key")
    print(f"Initial key: ...{initial[-4:]}")

    # Rotate
    secret_manager.rotate("github_api_key", "ghp_new_rotated_key_xxx")

    # After rotation
    rotated = secret_manager.get("github_api_key")
    print(f"After rotation: ...{rotated[-4:]}")

    # Check version
    secret = secret_manager.secrets["github_api_key"]
    print(f"Current version: {secret.version}")


# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ CX-01: AUTH & CREDENTIAL MANAGEMENT - EVALUATION SUMMARY                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LangGraph Native Support: ⭐ (Not Supported)                                │
│                                                                             │
│ LangGraph does NOT provide:                                                 │
│   ❌ Secret management                                                      │
│   ❌ OAuth client                                                           │
│   ❌ Token rotation                                                         │
│   ❌ Credential injection                                                   │
│   ❌ Service authentication                                                 │
│                                                                             │
│ Custom Implementation Required:                                             │
│   ✅ SecretManager - Store, retrieve, rotate secrets                        │
│   ✅ OAuthClient - OAuth 2.0 flow with refresh                              │
│   ✅ CredentialProvider - Just-in-time credential injection                 │
│   ✅ AuthenticatedToolNode - Tool wrapper with auth                         │
│                                                                             │
│ Features Implemented:                                                       │
│   ✓ Secret storage with expiration                                          │
│   ✓ Secret versioning                                                       │
│   ✓ OAuth 2.0 authorization code flow                                       │
│   ✓ Token refresh                                                           │
│   ✓ Token revocation                                                        │
│   ✓ API key rotation                                                        │
│   ✓ Tool-to-credential mapping                                              │
│                                                                             │
│ Production Considerations:                                                  │
│   - Use managed secret service (AWS Secrets Manager, Vault)                 │
│   - Encrypt secrets at rest                                                 │
│   - Implement audit logging for secret access                               │
│   - Use short-lived tokens                                                  │
│   - Implement token revocation on security events                           │
│   - Consider mTLS for service-to-service auth                               │
│                                                                             │
│ Rating: ⭐ (Not Supported - fully custom)                                   │
│   - No native auth support                                                  │
│   - Pattern works but significant implementation                            │
│   - Production needs managed secret service                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_secret_management()
    test_oauth_flow()
    test_credential_provider()
    test_token_rotation()
    test_authenticated_execution()

    print(SUMMARY)
