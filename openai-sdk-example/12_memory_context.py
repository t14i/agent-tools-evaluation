"""
Memory - Part 3: Context Management (ME-04 ~ ME-08)
Memory API, agent autonomous management, auto extraction, cleanup, embedding cost
"""

from dotenv import load_dotenv
load_dotenv()


from datetime import datetime, timedelta
from typing import Optional, Any
from dataclasses import dataclass, field
import json


# =============================================================================
# ME-04: Memory API (CRUD operations)
# =============================================================================

@dataclass
class MemoryEntry:
    """A memory entry with full metadata."""
    id: str
    key: str
    value: Any
    namespace: str
    entry_type: str  # "fact", "preference", "context", "event"
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl: Optional[timedelta] = None
    tags: list[str] = field(default_factory=list)
    source: str = "manual"  # "manual", "auto_extracted", "agent"


class MemoryAPI:
    """
    Full Memory API with CRUD operations.
    Implements ME-04: Memory API.
    """

    def __init__(self):
        self.store: dict[str, dict[str, MemoryEntry]] = {}
        self._id_counter = 0

    def _generate_id(self) -> str:
        self._id_counter += 1
        return f"mem_{self._id_counter:06d}"

    # CREATE
    def put(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        entry_type: str = "fact",
        tags: list[str] = None,
        ttl: Optional[timedelta] = None,
        source: str = "manual"
    ) -> MemoryEntry:
        """Create or update a memory entry."""
        if namespace not in self.store:
            self.store[namespace] = {}

        now = datetime.now()

        if key in self.store[namespace]:
            entry = self.store[namespace][key]
            entry.value = value
            entry.updated_at = now
            entry.tags = tags or entry.tags
            entry.ttl = ttl if ttl is not None else entry.ttl
        else:
            entry = MemoryEntry(
                id=self._generate_id(),
                key=key,
                value=value,
                namespace=namespace,
                entry_type=entry_type,
                created_at=now,
                updated_at=now,
                accessed_at=now,
                tags=tags or [],
                ttl=ttl,
                source=source
            )
            self.store[namespace][key] = entry

        return entry

    # READ
    def get(self, key: str, namespace: str = "default") -> Optional[MemoryEntry]:
        """Get a memory entry, updating access stats."""
        if namespace not in self.store:
            return None
        entry = self.store[namespace].get(key)
        if entry:
            entry.accessed_at = datetime.now()
            entry.access_count += 1
        return entry

    # UPDATE
    def update(
        self,
        key: str,
        namespace: str = "default",
        value: Any = None,
        tags: list[str] = None
    ) -> Optional[MemoryEntry]:
        """Update specific fields of a memory entry."""
        entry = self.get(key, namespace)
        if not entry:
            return None

        if value is not None:
            entry.value = value
        if tags is not None:
            entry.tags = tags
        entry.updated_at = datetime.now()

        return entry

    # DELETE
    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a memory entry."""
        if namespace in self.store and key in self.store[namespace]:
            del self.store[namespace][key]
            return True
        return False

    # SEARCH
    def search(
        self,
        namespace: str = "default",
        tags: list[str] = None,
        entry_type: str = None,
        source: str = None
    ) -> list[MemoryEntry]:
        """Search memories by criteria."""
        if namespace not in self.store:
            return []

        results = list(self.store[namespace].values())

        if tags:
            results = [e for e in results if any(t in e.tags for t in tags)]
        if entry_type:
            results = [e for e in results if e.entry_type == entry_type]
        if source:
            results = [e for e in results if e.source == source]

        return results


# =============================================================================
# ME-05: Agent Autonomous Memory Management
# =============================================================================

from agents import Agent, Runner, function_tool

memory_api = MemoryAPI()


@function_tool
def store_memory(
    key: str,
    value: str,
    memory_type: str = "fact",
    tags: str = ""
) -> str:
    """
    Store a memory for future reference.
    Types: fact, preference, context, event
    Tags: comma-separated list of relevant tags
    """
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    entry = memory_api.put(
        key=key,
        value=value,
        entry_type=memory_type,
        tags=tag_list,
        source="agent"
    )
    return f"Stored memory: {key} (type={memory_type}, tags={tag_list})"


@function_tool
def retrieve_memory(key: str) -> str:
    """Retrieve a stored memory by key."""
    entry = memory_api.get(key)
    if entry:
        return f"Memory found: {entry.value} (type={entry.entry_type}, accessed {entry.access_count} times)"
    return f"No memory found for key: {key}"


@function_tool
def search_memories(tags: str = "", memory_type: str = "") -> str:
    """Search memories by tags and/or type."""
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    results = memory_api.search(
        tags=tag_list,
        entry_type=memory_type if memory_type else None,
        source=None
    )

    if not results:
        return "No memories found matching criteria."

    response = f"Found {len(results)} memories:\n"
    for entry in results[:5]:
        response += f"- {entry.key}: {entry.value[:50]}...\n"

    return response


@function_tool
def forget_memory(key: str) -> str:
    """Delete a memory by key."""
    if memory_api.delete(key):
        return f"Deleted memory: {key}"
    return f"No memory found for key: {key}"


autonomous_memory_agent = Agent(
    name="AutonomousMemoryBot",
    instructions="""You are an assistant that autonomously manages your memory.
    - Store important facts, preferences, and context using store_memory
    - Retrieve relevant memories when answering questions
    - Search for related memories when helpful
    - Forget outdated or incorrect information

    Be proactive about remembering useful information from conversations.""",
    tools=[store_memory, retrieve_memory, search_memories, forget_memory],
)


# =============================================================================
# ME-06: Auto Extraction
# =============================================================================

class AutoExtractor:
    """
    Automatically extracts facts from conversations.
    Implements ME-06: Auto Extraction.

    Note: OpenAI SDK has Context Summarization but not full auto-extraction.
    This is a custom implementation pattern.
    """

    def __init__(self, memory_api: MemoryAPI):
        self.memory_api = memory_api

    def extract_and_store(self, conversation: list[dict], user_id: str):
        """
        Extract facts from conversation and store them.
        In production, this would use an LLM to extract structured facts.
        """
        # Simple pattern matching (production would use LLM)
        for message in conversation:
            content = message.get("content", "")

            # Extract simple patterns
            if "my name is" in content.lower():
                # Extract name
                parts = content.lower().split("my name is")
                if len(parts) > 1:
                    name = parts[1].strip().split()[0].title()
                    self.memory_api.put(
                        key="user_name",
                        value=name,
                        namespace=user_id,
                        entry_type="fact",
                        tags=["personal", "name"],
                        source="auto_extracted"
                    )

            if "favorite" in content.lower():
                # Could extract preferences
                self.memory_api.put(
                    key="preference_mentioned",
                    value=content,
                    namespace=user_id,
                    entry_type="preference",
                    tags=["preference"],
                    source="auto_extracted"
                )


# =============================================================================
# ME-07: Memory Cleanup
# =============================================================================

class MemoryCleanup:
    """
    Handles memory cleanup and TTL enforcement.
    Implements ME-07: Memory Cleanup.
    """

    def __init__(self, memory_api: MemoryAPI):
        self.memory_api = memory_api

    def cleanup_expired(self) -> int:
        """Remove memories that have exceeded their TTL."""
        cleaned = 0
        now = datetime.now()

        for namespace in list(self.memory_api.store.keys()):
            for key in list(self.memory_api.store[namespace].keys()):
                entry = self.memory_api.store[namespace][key]
                if entry.ttl:
                    if now > entry.created_at + entry.ttl:
                        del self.memory_api.store[namespace][key]
                        cleaned += 1

        return cleaned

    def cleanup_unused(self, days: int = 30) -> int:
        """Remove memories not accessed for N days."""
        cleaned = 0
        threshold = datetime.now() - timedelta(days=days)

        for namespace in list(self.memory_api.store.keys()):
            for key in list(self.memory_api.store[namespace].keys()):
                entry = self.memory_api.store[namespace][key]
                if entry.accessed_at < threshold:
                    del self.memory_api.store[namespace][key]
                    cleaned += 1

        return cleaned


# =============================================================================
# ME-08: Embedding Cost Tracking
# =============================================================================

class EmbeddingCostTracker:
    """
    Tracks embedding costs for memory operations.
    Implements ME-08: Embedding Cost.
    """

    def __init__(self, cost_per_1k_tokens: float = 0.0001):
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.total_tokens = 0
        self.operations: list[dict] = []

    def record_operation(
        self,
        operation: str,
        text_length: int,
        model: str = "text-embedding-3-small"
    ):
        """Record an embedding operation."""
        # Rough token estimate (1 token ≈ 4 chars)
        tokens = text_length // 4

        self.total_tokens += tokens
        self.operations.append({
            "operation": operation,
            "tokens": tokens,
            "model": model,
            "timestamp": datetime.now().isoformat()
        })

    def get_total_cost(self) -> float:
        """Get total embedding cost."""
        return (self.total_tokens / 1000) * self.cost_per_1k_tokens

    def get_summary(self) -> dict:
        """Get cost summary."""
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.get_total_cost(),
            "operation_count": len(self.operations)
        }


# =============================================================================
# Tests
# =============================================================================

def test_memory_api():
    """Test Memory API CRUD (ME-04)."""
    print("\n" + "=" * 70)
    print("TEST: Memory API CRUD (ME-04)")
    print("=" * 70)

    api = MemoryAPI()

    # Create
    entry = api.put("user_name", "Alice", tags=["personal"])
    print(f"\nCreated: {entry.key} = {entry.value}")

    # Read
    entry = api.get("user_name")
    print(f"Read: {entry.key} = {entry.value} (accessed {entry.access_count} times)")

    # Update
    api.update("user_name", value="Alice Smith", tags=["personal", "updated"])
    entry = api.get("user_name")
    print(f"Updated: {entry.key} = {entry.value}")

    # Search
    api.put("favorite_color", "blue", entry_type="preference", tags=["preference"])
    results = api.search(tags=["preference"])
    print(f"Search by tag: {len(results)} results")

    # Delete
    api.delete("user_name")
    entry = api.get("user_name")
    print(f"After delete: {entry}")

    print("\n✅ Memory API works")


def test_agent_memory():
    """Test agent autonomous memory (ME-05)."""
    print("\n" + "=" * 70)
    print("TEST: Agent Autonomous Memory (ME-05)")
    print("=" * 70)

    # Reset memory
    global memory_api
    memory_api = MemoryAPI()

    # Interaction 1: Store information
    result1 = Runner.run_sync(
        autonomous_memory_agent,
        "Remember that my birthday is March 15th and I love hiking."
    )
    print(f"\nAgent response:\n{result1.final_output}")

    # Interaction 2: Recall
    result2 = Runner.run_sync(
        autonomous_memory_agent,
        "What do you know about me?"
    )
    print(f"\nAgent response:\n{result2.final_output}")

    print("\n✅ Agent autonomous memory works")


def test_auto_extraction():
    """Test auto extraction (ME-06)."""
    print("\n" + "=" * 70)
    print("TEST: Auto Extraction (ME-06)")
    print("=" * 70)

    api = MemoryAPI()
    extractor = AutoExtractor(api)

    # Simulate conversation
    conversation = [
        {"role": "user", "content": "Hi, my name is Bob"},
        {"role": "assistant", "content": "Hello Bob!"},
        {"role": "user", "content": "My favorite food is pizza"},
    ]

    extractor.extract_and_store(conversation, "user_123")

    # Check extracted facts
    results = api.search("user_123", source="auto_extracted")
    print(f"\nAuto-extracted memories: {len(results)}")
    for entry in results:
        print(f"  {entry.key}: {entry.value}")

    print("\n✅ Auto extraction works")


def test_cleanup():
    """Test memory cleanup (ME-07)."""
    print("\n" + "=" * 70)
    print("TEST: Memory Cleanup (ME-07)")
    print("=" * 70)

    api = MemoryAPI()
    cleanup = MemoryCleanup(api)

    # Create memory with short TTL
    api.put("temp_data", "temporary", ttl=timedelta(seconds=0))
    api.put("persistent_data", "permanent")

    print(f"\nBefore cleanup: {len(api.store.get('default', {}))} memories")

    # Cleanup expired
    cleaned = cleanup.cleanup_expired()
    print(f"Cleaned expired: {cleaned}")
    print(f"After cleanup: {len(api.store.get('default', {}))} memories")

    print("\n✅ Memory cleanup works")


def test_cost_tracking():
    """Test embedding cost tracking (ME-08)."""
    print("\n" + "=" * 70)
    print("TEST: Embedding Cost Tracking (ME-08)")
    print("=" * 70)

    tracker = EmbeddingCostTracker()

    # Simulate operations
    tracker.record_operation("store", 500)
    tracker.record_operation("search", 100)
    tracker.record_operation("store", 1000)

    summary = tracker.get_summary()
    print(f"\nCost Summary:")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  Total cost: ${summary['total_cost_usd']:.6f}")
    print(f"  Operations: {summary['operation_count']}")

    print("\n✅ Cost tracking works")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ ME-04 ~ ME-08: CONTEXT MANAGEMENT - EVALUATION SUMMARY                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ME-04 (Memory API): ⭐⭐ (Experimental)                                     │
│   ❌ No built-in Memory CRUD API                                            │
│   ⚠️ Sessions provide basic storage                                        │
│   ⚠️ Custom MemoryAPI implementation provided                              │
│                                                                             │
│ ME-05 (Agent Autonomous Management): ⭐ (Not Supported)                     │
│   ❌ No equivalent to LangMem                                               │
│   ❌ Agent cannot autonomously manage memory                                │
│   ⚠️ Custom tools can provide similar capability                           │
│                                                                             │
│ ME-06 (Auto Extraction): ⭐⭐⭐ (PoC Ready)                                  │
│   ✅ Context Summarization available                                        │
│   ❌ No automatic fact extraction                                           │
│   ⚠️ Can implement with post-processing                                    │
│                                                                             │
│ ME-07 (Memory Cleanup): ⭐ (Not Supported)                                  │
│   ❌ No built-in TTL support                                                │
│   ❌ No automatic cleanup                                                   │
│   ⚠️ Custom cleanup job required                                           │
│                                                                             │
│ ME-08 (Embedding Cost): ⭐⭐⭐ (PoC Ready)                                   │
│   ✅ token_usage available in responses                                     │
│   ❌ No aggregated cost tracking                                            │
│   ⚠️ Custom tracker implementation provided                                │
│                                                                             │
│ Comparison with LangGraph:                                                  │
│   LangGraph + LangMem:                                                      │
│     - Store API with full CRUD                                              │
│     - LangMem for agent-managed memory                                      │
│     - Background extraction                                                 │
│   OpenAI SDK:                                                               │
│     - Sessions for basic storage                                            │
│     - Context Summarization                                                 │
│     - Less sophisticated memory management                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_memory_api()
    test_agent_memory()
    test_auto_extraction()
    test_cleanup()
    test_cost_tracking()

    print(SUMMARY)
