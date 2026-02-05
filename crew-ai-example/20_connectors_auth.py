"""
20_connectors_auth.py - Connectors Authentication (CX-01)

Purpose: Verify authentication and credential management
- OAuth token refresh/rotation
- Secret management integration
- Secure credential handling

LangGraph Comparison:
- Both require custom implementation for auth management
- No built-in credential management in either framework
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Type, Optional, Any
from enum import Enum

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# =============================================================================
# Token Management
# =============================================================================

class TokenType(str, Enum):
    """Types of authentication tokens."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    JWT = "jwt"


class Token(BaseModel):
    """Authentication token with metadata."""

    token_type: TokenType
    value: str
    expires_at: Optional[datetime] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at

    @property
    def needs_refresh(self) -> bool:
        """Check if token needs refresh (within 5 min of expiry)."""
        if self.expires_at is None:
            return False
        buffer = timedelta(minutes=5)
        return datetime.now() >= (self.expires_at - buffer)


class TokenStore:
    """
    Secure token storage with automatic refresh.

    In production, would use encrypted storage or vault.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./db/tokens")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.tokens: dict[str, Token] = {}
        self._encryption_key: Optional[bytes] = None

    def store(self, service_name: str, token: Token):
        """Store a token for a service."""
        self.tokens[service_name] = token
        self._persist(service_name)
        print(f"[TokenStore] Stored token for {service_name}")

    def get(self, service_name: str) -> Optional[Token]:
        """Get a token for a service."""
        if service_name not in self.tokens:
            self._load(service_name)
        return self.tokens.get(service_name)

    def _persist(self, service_name: str):
        """Persist token to storage (encrypted in production)."""
        if service_name not in self.tokens:
            return

        token = self.tokens[service_name]
        file_path = self.storage_path / f"{service_name}.json"

        # In production, would encrypt before saving
        data = {
            "token_type": token.token_type.value,
            "value": token.value[:8] + "...[REDACTED]",  # Redact for demo
            "expires_at": token.expires_at.isoformat() if token.expires_at else None,
            "scope": token.scope,
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self, service_name: str):
        """Load token from storage."""
        file_path = self.storage_path / f"{service_name}.json"
        if not file_path.exists():
            return

        with open(file_path, "r") as f:
            data = json.load(f)

        # In production, would decrypt after loading
        # For demo, we can't restore the full token from redacted version


# =============================================================================
# OAuth Client
# =============================================================================

class OAuthConfig(BaseModel):
    """OAuth client configuration."""

    client_id: str
    client_secret: str
    auth_url: str
    token_url: str
    scope: str = ""
    redirect_uri: str = ""


class OAuthClient:
    """
    OAuth 2.0 client with automatic token refresh.

    Supports:
    - Authorization Code flow
    - Client Credentials flow
    - Token refresh
    """

    def __init__(self, config: OAuthConfig, token_store: TokenStore):
        self.config = config
        self.token_store = token_store
        self.service_name = f"oauth_{config.client_id[:8]}"

    def get_access_token(self) -> Optional[str]:
        """Get valid access token, refreshing if needed."""
        token = self.token_store.get(self.service_name)

        if token is None:
            print(f"[OAuth] No token found for {self.service_name}")
            return None

        if token.needs_refresh:
            print(f"[OAuth] Token needs refresh")
            if token.refresh_token:
                return self._refresh_token(token)
            else:
                print(f"[OAuth] No refresh token available")
                return None

        return token.value

    def _refresh_token(self, token: Token) -> Optional[str]:
        """Refresh an expired token."""
        print(f"[OAuth] Refreshing token...")

        # In production, would make actual HTTP request
        # Simulating token refresh
        new_token = Token(
            token_type=TokenType.ACCESS,
            value=f"new_access_token_{int(time.time())}",
            expires_at=datetime.now() + timedelta(hours=1),
            refresh_token=f"new_refresh_token_{int(time.time())}",
            scope=token.scope,
        )

        self.token_store.store(self.service_name, new_token)
        print(f"[OAuth] Token refreshed successfully")
        return new_token.value

    def authenticate_client_credentials(self) -> Optional[str]:
        """Authenticate using client credentials flow."""
        print(f"[OAuth] Authenticating with client credentials...")

        # In production, would make actual HTTP request
        # Simulating authentication
        token = Token(
            token_type=TokenType.ACCESS,
            value=f"client_credentials_token_{int(time.time())}",
            expires_at=datetime.now() + timedelta(hours=1),
            scope=self.config.scope,
        )

        self.token_store.store(self.service_name, token)
        return token.value


# =============================================================================
# Secret Manager
# =============================================================================

class SecretSource(str, Enum):
    """Sources for secrets."""
    ENV = "environment"
    FILE = "file"
    VAULT = "vault"
    AWS_SECRETS = "aws_secrets_manager"
    GCP_SECRETS = "gcp_secret_manager"
    AZURE_KEYVAULT = "azure_keyvault"


class SecretManager:
    """
    Centralized secret management.

    Supports multiple sources:
    - Environment variables
    - Config files (encrypted)
    - Cloud secret managers (simulated)
    """

    def __init__(self, default_source: SecretSource = SecretSource.ENV):
        self.default_source = default_source
        self._cache: dict[str, Any] = {}
        self._sources: dict[SecretSource, callable] = {
            SecretSource.ENV: self._get_from_env,
            SecretSource.FILE: self._get_from_file,
            SecretSource.VAULT: self._get_from_vault,
            SecretSource.AWS_SECRETS: self._get_from_aws,
        }

    def get(
        self,
        key: str,
        source: Optional[SecretSource] = None,
        required: bool = True,
    ) -> Optional[str]:
        """Get a secret value."""
        cache_key = f"{source or self.default_source}:{key}"

        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get from source
        source = source or self.default_source
        getter = self._sources.get(source)

        if getter is None:
            raise ValueError(f"Unknown secret source: {source}")

        value = getter(key)

        if value is None and required:
            raise ValueError(f"Required secret not found: {key}")

        # Cache the value
        if value is not None:
            self._cache[cache_key] = value

        return value

    def _get_from_env(self, key: str) -> Optional[str]:
        """Get secret from environment variable."""
        value = os.environ.get(key)
        if value:
            print(f"[SecretManager] Loaded {key} from environment")
        return value

    def _get_from_file(self, key: str) -> Optional[str]:
        """Get secret from encrypted file."""
        secrets_file = Path("./db/secrets.json")
        if not secrets_file.exists():
            return None

        with open(secrets_file, "r") as f:
            secrets = json.load(f)

        value = secrets.get(key)
        if value:
            print(f"[SecretManager] Loaded {key} from file")
        return value

    def _get_from_vault(self, key: str) -> Optional[str]:
        """Get secret from HashiCorp Vault (simulated)."""
        # In production, would use hvac library
        print(f"[SecretManager] Would fetch {key} from Vault")
        return f"vault_secret_{key}"

    def _get_from_aws(self, key: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager (simulated)."""
        # In production, would use boto3
        print(f"[SecretManager] Would fetch {key} from AWS Secrets Manager")
        return f"aws_secret_{key}"

    def clear_cache(self):
        """Clear the secret cache."""
        self._cache.clear()


# =============================================================================
# Authenticated Tool
# =============================================================================

class AuthenticatedAPIInput(BaseModel):
    """Input for authenticated API tool."""

    endpoint: str = Field(..., description="API endpoint to call")
    method: str = Field(default="GET", description="HTTP method")
    data: Optional[dict] = Field(default=None, description="Request body")


class AuthenticatedAPITool(BaseTool):
    """
    API tool with automatic authentication.

    Handles token refresh and retry on auth failures.
    """

    name: str = "Authenticated API"
    description: str = """Make authenticated API calls.
    Handles OAuth token refresh automatically."""
    args_schema: Type[BaseModel] = AuthenticatedAPIInput

    oauth_client: Optional[OAuthClient] = None
    secret_manager: Optional[SecretManager] = None
    api_key_name: str = "API_KEY"

    def __init__(
        self,
        oauth_client: Optional[OAuthClient] = None,
        secret_manager: Optional[SecretManager] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.oauth_client = oauth_client
        self.secret_manager = secret_manager or SecretManager()

    def _get_auth_header(self) -> Optional[str]:
        """Get authentication header value."""
        # Try OAuth first
        if self.oauth_client:
            token = self.oauth_client.get_access_token()
            if token:
                return f"Bearer {token}"

        # Fall back to API key
        try:
            api_key = self.secret_manager.get(self.api_key_name, required=False)
            if api_key:
                return f"ApiKey {api_key}"
        except Exception:
            pass

        return None

    def _run(self, endpoint: str, method: str = "GET", data: Optional[dict] = None) -> str:
        """Execute authenticated API call."""
        auth_header = self._get_auth_header()

        if not auth_header:
            return "[ERROR] No authentication available"

        print(f"  [AuthAPI] {method} {endpoint}")
        print(f"  [AuthAPI] Auth: {auth_header[:20]}...")

        # Simulate API call
        return f"[SUCCESS] {method} {endpoint} - Authenticated"


# =============================================================================
# Credential Rotation
# =============================================================================

class CredentialRotator:
    """
    Handles credential rotation.

    Ensures smooth transitions during credential updates.
    """

    def __init__(self, token_store: TokenStore):
        self.token_store = token_store
        self.rotation_log: list[dict] = []

    def rotate_token(
        self,
        service_name: str,
        new_token: Token,
        grace_period_seconds: int = 300,
    ) -> bool:
        """
        Rotate a token with grace period.

        During grace period, both old and new tokens are valid.
        """
        old_token = self.token_store.get(service_name)

        rotation_entry = {
            "service": service_name,
            "timestamp": datetime.now().isoformat(),
            "old_expires_at": old_token.expires_at.isoformat() if old_token and old_token.expires_at else None,
            "new_expires_at": new_token.expires_at.isoformat() if new_token.expires_at else None,
            "grace_period": grace_period_seconds,
        }

        print(f"[Rotation] Rotating token for {service_name}")
        print(f"[Rotation] Grace period: {grace_period_seconds}s")

        # Store new token
        self.token_store.store(service_name, new_token)

        # Log rotation
        rotation_entry["status"] = "completed"
        self.rotation_log.append(rotation_entry)

        return True

    def get_rotation_history(self, service_name: Optional[str] = None) -> list[dict]:
        """Get rotation history."""
        if service_name:
            return [r for r in self.rotation_log if r["service"] == service_name]
        return self.rotation_log


# =============================================================================
# Demonstrations
# =============================================================================

def demo_token_management():
    """Demonstrate token management."""
    print("=" * 60)
    print("Demo 1: Token Management")
    print("=" * 60)

    store = TokenStore()

    # Create and store a token
    token = Token(
        token_type=TokenType.ACCESS,
        value="access_token_12345",
        expires_at=datetime.now() + timedelta(hours=1),
        refresh_token="refresh_token_67890",
        scope="read write",
    )

    store.store("demo_service", token)

    # Retrieve token
    retrieved = store.get("demo_service")
    if retrieved:
        print(f"\nToken retrieved:")
        print(f"  Type: {retrieved.token_type.value}")
        print(f"  Expired: {retrieved.is_expired}")
        print(f"  Needs refresh: {retrieved.needs_refresh}")

    # Create expired token
    expired_token = Token(
        token_type=TokenType.ACCESS,
        value="old_token",
        expires_at=datetime.now() - timedelta(hours=1),  # Already expired
        refresh_token="refresh_for_old",
    )

    store.store("expired_service", expired_token)
    exp = store.get("expired_service")
    if exp:
        print(f"\nExpired token check:")
        print(f"  Is expired: {exp.is_expired}")


def demo_oauth_flow():
    """Demonstrate OAuth authentication."""
    print("\n" + "=" * 60)
    print("Demo 2: OAuth Authentication")
    print("=" * 60)

    config = OAuthConfig(
        client_id="demo_client_id",
        client_secret="demo_client_secret",
        auth_url="https://auth.example.com/authorize",
        token_url="https://auth.example.com/token",
        scope="read write",
    )

    store = TokenStore()
    oauth = OAuthClient(config, store)

    # Authenticate with client credentials
    print("\n--- Client Credentials Flow ---")
    token = oauth.authenticate_client_credentials()
    print(f"Obtained token: {token[:20]}...")

    # Get token (should use cached)
    print("\n--- Getting Cached Token ---")
    token2 = oauth.get_access_token()
    print(f"Token: {token2[:20]}..." if token2 else "No token")


def demo_secret_manager():
    """Demonstrate secret management."""
    print("\n" + "=" * 60)
    print("Demo 3: Secret Management")
    print("=" * 60)

    manager = SecretManager()

    # Set up test environment variable
    os.environ["TEST_API_KEY"] = "test_key_12345"

    # Get from environment
    print("\n--- Environment Source ---")
    key = manager.get("TEST_API_KEY", source=SecretSource.ENV)
    print(f"API Key: {key}")

    # Get from vault (simulated)
    print("\n--- Vault Source (simulated) ---")
    secret = manager.get("DATABASE_PASSWORD", source=SecretSource.VAULT)
    print(f"DB Password: {secret}")

    # Clean up
    del os.environ["TEST_API_KEY"]


def demo_authenticated_tool():
    """Demonstrate authenticated API tool."""
    print("\n" + "=" * 60)
    print("Demo 4: Authenticated API Tool")
    print("=" * 60)

    # Set up OAuth
    config = OAuthConfig(
        client_id="demo_client",
        client_secret="demo_secret",
        auth_url="https://auth.example.com/authorize",
        token_url="https://auth.example.com/token",
    )

    store = TokenStore()
    oauth = OAuthClient(config, store)
    oauth.authenticate_client_credentials()

    # Create tool with OAuth
    tool = AuthenticatedAPITool(oauth_client=oauth)

    # Make API calls
    print("\n--- Making Authenticated Calls ---")
    result = tool._run(endpoint="/api/users", method="GET")
    print(f"Result: {result}")

    result = tool._run(endpoint="/api/data", method="POST", data={"key": "value"})
    print(f"Result: {result}")


def demo_credential_rotation():
    """Demonstrate credential rotation."""
    print("\n" + "=" * 60)
    print("Demo 5: Credential Rotation")
    print("=" * 60)

    store = TokenStore()
    rotator = CredentialRotator(store)

    # Store initial token
    old_token = Token(
        token_type=TokenType.ACCESS,
        value="old_token_value",
        expires_at=datetime.now() + timedelta(hours=1),
    )
    store.store("rotation_demo", old_token)

    # Rotate to new token
    new_token = Token(
        token_type=TokenType.ACCESS,
        value="new_token_value",
        expires_at=datetime.now() + timedelta(hours=2),
    )

    rotator.rotate_token("rotation_demo", new_token, grace_period_seconds=60)

    # Check rotation history
    history = rotator.get_rotation_history("rotation_demo")
    print("\nRotation history:")
    for entry in history:
        print(f"  {entry['timestamp']}: {entry['status']}")


def demo_multi_source_secrets():
    """Demonstrate secrets from multiple sources."""
    print("\n" + "=" * 60)
    print("Demo 6: Multi-Source Secret Management")
    print("=" * 60)

    manager = SecretManager()

    # Create secrets file
    secrets_dir = Path("./db")
    secrets_dir.mkdir(exist_ok=True)

    with open(secrets_dir / "secrets.json", "w") as f:
        json.dump({
            "FILE_SECRET": "secret_from_file",
        }, f)

    # Get from different sources
    print("\n--- Getting secrets from different sources ---")

    os.environ["ENV_SECRET"] = "secret_from_env"
    env_secret = manager.get("ENV_SECRET", source=SecretSource.ENV)
    print(f"From ENV: {env_secret}")
    del os.environ["ENV_SECRET"]

    file_secret = manager.get("FILE_SECRET", source=SecretSource.FILE)
    print(f"From FILE: {file_secret}")

    vault_secret = manager.get("VAULT_SECRET", source=SecretSource.VAULT)
    print(f"From VAULT: {vault_secret}")


def main():
    print("=" * 60)
    print("Connectors Authentication Verification (CX-01)")
    print("=" * 60)
    print("""
This script verifies authentication and credential management.

Verification Items:
- OAuth token refresh/rotation
- Secret management from multiple sources
- Secure credential handling
- Automatic token refresh in tools

Key Components:
- TokenStore: Secure token storage
- OAuthClient: OAuth 2.0 authentication
- SecretManager: Multi-source secret retrieval
- CredentialRotator: Safe credential rotation

Secret Sources:
- Environment variables
- Encrypted config files
- HashiCorp Vault
- Cloud secret managers (AWS, GCP, Azure)

LangGraph Comparison:
- Neither has built-in auth management
- Both require custom implementation
- Similar patterns apply to both frameworks
""")

    # Run all demos
    demo_token_management()
    demo_oauth_flow()
    demo_secret_manager()
    demo_authenticated_tool()
    demo_credential_rotation()
    demo_multi_source_secrets()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
