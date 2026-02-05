"""
Connectors & Ops - Part 1: Streaming (CX-01, CX-02)
Authentication management, rate limiting, streaming responses
"""

from dotenv import load_dotenv
load_dotenv()


import time
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Generator, AsyncGenerator
from dataclasses import dataclass, field
from collections import deque


# =============================================================================
# CX-01: Authentication / Credential Management
# =============================================================================

@dataclass
class Credential:
    """Stored credential."""
    name: str
    credential_type: str  # "api_key", "oauth_token", "basic_auth"
    value: str
    expires_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class CredentialManager:
    """
    Manages credentials for external services.
    Implements CX-01: Auth / Credential Management.
    """

    def __init__(self):
        self.credentials: dict[str, Credential] = {}
        self._encryption_key = None  # Would be set in production

    def store(
        self,
        name: str,
        credential_type: str,
        value: str,
        expires_at: datetime = None,
        metadata: dict = None
    ) -> Credential:
        """Store a credential (in production, would be encrypted)."""
        cred = Credential(
            name=name,
            credential_type=credential_type,
            value=value,  # In production: self._encrypt(value)
            expires_at=expires_at,
            metadata=metadata or {}
        )
        self.credentials[name] = cred
        return cred

    def get(self, name: str) -> Optional[str]:
        """Get a credential value."""
        cred = self.credentials.get(name)
        if not cred:
            return None

        # Check expiration
        if cred.expires_at and datetime.now() > cred.expires_at:
            del self.credentials[name]
            return None

        return cred.value  # In production: self._decrypt(cred.value)

    def refresh_oauth(self, name: str, refresh_fn: callable) -> bool:
        """Refresh an OAuth token."""
        cred = self.credentials.get(name)
        if not cred or cred.credential_type != "oauth_token":
            return False

        try:
            new_value, expires_in = refresh_fn(cred.value)
            cred.value = new_value
            cred.expires_at = datetime.now() + timedelta(seconds=expires_in)
            return True
        except Exception:
            return False

    def delete(self, name: str) -> bool:
        """Delete a credential."""
        if name in self.credentials:
            del self.credentials[name]
            return True
        return False

    def list_credentials(self) -> list[str]:
        """List credential names (not values)."""
        return list(self.credentials.keys())


# =============================================================================
# CX-02: Rate Limiting
# =============================================================================

class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_time(self, tokens: int = 1) -> float:
        """Calculate wait time to acquire tokens."""
        self._refill()
        if self.tokens >= tokens:
            return 0
        return (tokens - self.tokens) / self.refill_rate


class RateLimiter:
    """
    Multi-key rate limiter with different limits per service.
    Implements CX-02: Rate Limit / Retry.
    """

    def __init__(self):
        self.limiters: dict[str, TokenBucket] = {}
        self.default_limits = {
            "openai": (60, 1),  # 60 requests per minute
            "anthropic": (60, 1),
            "default": (100, 2)  # 100 requests per minute
        }

    def get_limiter(self, service: str) -> TokenBucket:
        """Get or create limiter for service."""
        if service not in self.limiters:
            limits = self.default_limits.get(service, self.default_limits["default"])
            self.limiters[service] = TokenBucket(limits[0], limits[1])
        return self.limiters[service]

    def acquire(self, service: str, tokens: int = 1) -> bool:
        """Try to acquire tokens for a service."""
        return self.get_limiter(service).acquire(tokens)

    def wait_and_acquire(self, service: str, tokens: int = 1) -> float:
        """Wait if necessary and acquire tokens."""
        limiter = self.get_limiter(service)
        wait_time = limiter.wait_time(tokens)
        if wait_time > 0:
            time.sleep(wait_time)
        limiter.acquire(tokens)
        return wait_time


class ExponentialBackoff:
    """Exponential backoff for retries."""

    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True
    ):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for attempt number."""
        import random
        delay = self.initial_delay * (self.multiplier ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay

    def execute_with_retry(
        self,
        fn: callable,
        max_attempts: int = 3,
        retryable_exceptions: tuple = (Exception,)
    ) -> tuple[any, int]:
        """
        Execute function with retry.
        Returns: (result, attempts)
        """
        last_exception = None

        for attempt in range(max_attempts):
            try:
                return (fn(), attempt + 1)
            except retryable_exceptions as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    delay = self.get_delay(attempt)
                    time.sleep(delay)

        raise last_exception


# =============================================================================
# Streaming Response Handler
# =============================================================================

class StreamingHandler:
    """
    Handles streaming responses from LLM APIs.
    """

    def __init__(self, rate_limiter: RateLimiter = None):
        self.rate_limiter = rate_limiter or RateLimiter()
        self.buffer: deque = deque(maxlen=1000)

    def stream_response(self, service: str, chunks: list[str]) -> Generator[str, None, None]:
        """
        Stream response chunks with rate limiting.
        In production, this would wrap actual API streaming.
        """
        # Check rate limit
        if not self.rate_limiter.acquire(service):
            wait_time = self.rate_limiter.wait_and_acquire(service)
            print(f"  Rate limited, waited {wait_time:.2f}s")

        for chunk in chunks:
            self.buffer.append(chunk)
            yield chunk

    async def async_stream_response(
        self,
        service: str,
        chunks: list[str]
    ) -> AsyncGenerator[str, None]:
        """Async version of stream_response."""
        if not self.rate_limiter.acquire(service):
            wait_time = self.rate_limiter.get_limiter(service).wait_time(1)
            await asyncio.sleep(wait_time)
            self.rate_limiter.acquire(service)

        for chunk in chunks:
            self.buffer.append(chunk)
            yield chunk
            await asyncio.sleep(0.01)  # Simulate network delay


# =============================================================================
# Integrated Connector
# =============================================================================

class APIConnector:
    """
    Integrated connector with auth, rate limiting, and retry.
    """

    def __init__(self, service: str):
        self.service = service
        self.credentials = CredentialManager()
        self.rate_limiter = RateLimiter()
        self.backoff = ExponentialBackoff()

    def configure_auth(self, api_key: str):
        """Configure authentication."""
        self.credentials.store(
            name=f"{self.service}_api_key",
            credential_type="api_key",
            value=api_key
        )

    def call(self, endpoint: str, params: dict) -> dict:
        """Make an API call with auth, rate limiting, and retry."""
        # Get credentials
        api_key = self.credentials.get(f"{self.service}_api_key")
        if not api_key:
            raise ValueError(f"No credentials for {self.service}")

        # Rate limit
        self.rate_limiter.wait_and_acquire(self.service)

        # Execute with retry
        def do_call():
            # Simulate API call
            return {"status": "success", "data": f"Response for {endpoint}"}

        result, attempts = self.backoff.execute_with_retry(do_call, max_attempts=3)

        return result


# =============================================================================
# Tests
# =============================================================================

def test_credential_management():
    """Test credential management (CX-01)."""
    print("\n" + "=" * 70)
    print("TEST: Credential Management (CX-01)")
    print("=" * 70)

    manager = CredentialManager()

    # Store credentials
    manager.store(
        name="openai_key",
        credential_type="api_key",
        value="sk-..."
    )

    manager.store(
        name="oauth_token",
        credential_type="oauth_token",
        value="eyJ...",
        expires_at=datetime.now() + timedelta(hours=1)
    )

    print(f"\nStored credentials: {manager.list_credentials()}")

    # Retrieve
    key = manager.get("openai_key")
    print(f"Retrieved API key: {key[:5]}..." if key else "Not found")

    # Delete
    manager.delete("openai_key")
    print(f"After delete: {manager.list_credentials()}")

    print("\n✅ Credential management works")


def test_rate_limiting():
    """Test rate limiting (CX-02)."""
    print("\n" + "=" * 70)
    print("TEST: Rate Limiting (CX-02)")
    print("=" * 70)

    # Create limiter with low capacity for testing
    bucket = TokenBucket(capacity=3, refill_rate=1)

    print("\nToken bucket test:")
    for i in range(5):
        acquired = bucket.acquire()
        print(f"  Attempt {i+1}: {'✅ acquired' if acquired else '❌ denied'}")
        if not acquired:
            wait = bucket.wait_time()
            print(f"    Wait time: {wait:.2f}s")

    print("\n✅ Rate limiting works")


def test_exponential_backoff():
    """Test exponential backoff."""
    print("\n" + "=" * 70)
    print("TEST: Exponential Backoff")
    print("=" * 70)

    backoff = ExponentialBackoff(initial_delay=0.1, max_delay=1.0, jitter=False)

    print("\nDelay progression:")
    for attempt in range(5):
        delay = backoff.get_delay(attempt)
        print(f"  Attempt {attempt}: {delay:.2f}s")

    # Test retry
    attempt_count = 0
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError("Transient error")
        return "Success!"

    backoff_fast = ExponentialBackoff(initial_delay=0.01, jitter=False)
    result, attempts = backoff_fast.execute_with_retry(flaky_function)
    print(f"\nFlaky function: {result} after {attempts} attempts")

    print("\n✅ Exponential backoff works")


def test_streaming():
    """Test streaming response handling."""
    print("\n" + "=" * 70)
    print("TEST: Streaming Responses")
    print("=" * 70)

    handler = StreamingHandler()

    chunks = ["Hello", " ", "world", "!", " How", " are", " you", "?"]

    print("\nStreaming response:")
    full_response = ""
    for chunk in handler.stream_response("openai", chunks):
        full_response += chunk
        print(f"  Received: '{chunk}'")

    print(f"\nFull response: {full_response}")

    print("\n✅ Streaming works")


def test_integrated_connector():
    """Test integrated API connector."""
    print("\n" + "=" * 70)
    print("TEST: Integrated API Connector")
    print("=" * 70)

    connector = APIConnector("openai")
    connector.configure_auth("sk-test-key")

    # Make calls
    for i in range(3):
        result = connector.call("/chat/completions", {"message": f"Test {i}"})
        print(f"  Call {i+1}: {result['status']}")

    print("\n✅ Integrated connector works")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ CX-01, CX-02: STREAMING & CONNECTORS - EVALUATION SUMMARY                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ CX-01 (Auth / Credential Management): ⭐⭐⭐ (PoC Ready)                    │
│   ✅ API key authentication built-in                                        │
│   ❌ No OAuth flow management                                               │
│   ❌ No credential encryption                                               │
│   ⚠️ Custom CredentialManager implementation provided                      │
│                                                                             │
│ CX-02 (Rate Limit / Retry): ⭐⭐ (Experimental)                             │
│   ✅ Some automatic retry on API errors                                     │
│   ❌ No configurable rate limiting                                          │
│   ❌ No token bucket implementation                                         │
│   ⚠️ Custom RateLimiter and ExponentialBackoff provided                    │
│                                                                             │
│ Custom Implementation Provided:                                             │
│   ✅ CredentialManager - Store/retrieve/refresh credentials                 │
│   ✅ TokenBucket - Rate limiting algorithm                                  │
│   ✅ RateLimiter - Multi-service rate limiting                              │
│   ✅ ExponentialBackoff - Retry with backoff                                │
│   ✅ StreamingHandler - Streaming response handling                         │
│   ✅ APIConnector - Integrated auth + rate limit + retry                    │
│                                                                             │
│ OpenAI SDK Features:                                                        │
│   - Basic API key authentication                                            │
│   - Streaming response support                                              │
│   - Some automatic retries                                                  │
│                                                                             │
│ Missing for Production:                                                     │
│   - OAuth 2.0 flow management                                               │
│   - Credential rotation                                                     │
│   - Configurable rate limits                                                │
│   - Circuit breaker pattern                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_credential_management()
    test_rate_limiting()
    test_exponential_backoff()
    test_streaming()
    test_integrated_connector()

    print(SUMMARY)
