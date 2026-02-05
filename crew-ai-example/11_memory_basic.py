"""
11_memory_basic.py - Basic Memory Functionality (ME-01 to ME-08)

Purpose: Verify CrewAI's memory feature
- What gets saved with memory=True
- Behavior of short-term memory
- Memory within the same session
- ME-03: Semantic search capability
- ME-04: Memory CRUD API
- ME-05: Agent autonomous memory management
- ME-06: Automatic fact extraction
- ME-07: Cleanup and TTL
- ME-08: Embedding cost measurement

LangGraph Comparison:
- LangGraph: Explicit memory management in State
- CrewAI: Automatic memory enablement with memory=True
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field

from crewai import Agent, Task, Crew


# =============================================================================
# Enhanced Memory Management (ME-03 to ME-08)
# =============================================================================

@dataclass
class MemoryEntry:
    """A single memory entry."""

    id: str
    content: str
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_hours: Optional[int] = None
    source: str = "manual"  # manual, extracted, conversation


class EnhancedMemoryStore:
    """
    Enhanced memory store with semantic search and lifecycle management.

    Implements ME-03 through ME-08.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./db/enhanced_memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memories: dict[str, MemoryEntry] = {}
        self.embedding_cost_per_token = 0.0001  # $0.0001 per 1K tokens approx
        self.total_embedding_cost = 0.0
        self._load_memories()

    def _load_memories(self):
        """Load memories from storage."""
        memory_file = self.storage_path / "memories.json"
        if memory_file.exists():
            with open(memory_file, "r") as f:
                data = json.load(f)
                for mid, mdata in data.items():
                    mdata["created_at"] = datetime.fromisoformat(mdata["created_at"])
                    mdata["accessed_at"] = datetime.fromisoformat(mdata["accessed_at"])
                    self.memories[mid] = MemoryEntry(**mdata)

    def _save_memories(self):
        """Save memories to storage."""
        memory_file = self.storage_path / "memories.json"
        data = {}
        for mid, memory in self.memories.items():
            data[mid] = {
                "id": memory.id,
                "content": memory.content,
                "embedding": memory.embedding,
                "metadata": memory.metadata,
                "created_at": memory.created_at.isoformat(),
                "accessed_at": memory.accessed_at.isoformat(),
                "access_count": memory.access_count,
                "ttl_hours": memory.ttl_hours,
                "source": memory.source,
            }
        with open(memory_file, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_id(self) -> str:
        """Generate unique memory ID."""
        import uuid
        return f"mem_{uuid.uuid4().hex[:12]}"

    def _compute_embedding(self, text: str) -> tuple[list[float], float]:
        """
        Compute embedding for text (simulated).

        Returns (embedding, cost).
        In production, would call actual embedding API.
        """
        # Simulate embedding (in production, use OpenAI/sentence-transformers)
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = [b / 255.0 for b in hash_bytes[:128]]  # 128-dim fake embedding

        # Calculate cost (ME-08)
        tokens = len(text.split())
        cost = (tokens / 1000) * self.embedding_cost_per_token
        self.total_embedding_cost += cost

        return embedding, cost

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between embeddings."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # =========================================================================
    # CRUD Operations (ME-04)
    # =========================================================================

    def create(
        self,
        content: str,
        metadata: Optional[dict] = None,
        ttl_hours: Optional[int] = None,
        source: str = "manual",
    ) -> str:
        """Create a new memory entry."""
        memory_id = self._generate_id()
        embedding, cost = self._compute_embedding(content)

        memory = MemoryEntry(
            id=memory_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            ttl_hours=ttl_hours,
            source=source,
        )

        self.memories[memory_id] = memory
        self._save_memories()

        print(f"[Memory] Created {memory_id}: '{content[:50]}...' (cost: ${cost:.6f})")
        return memory_id

    def read(self, memory_id: str) -> Optional[MemoryEntry]:
        """Read a memory by ID."""
        if memory_id not in self.memories:
            return None

        memory = self.memories[memory_id]
        memory.accessed_at = datetime.now()
        memory.access_count += 1
        self._save_memories()

        return memory

    def update(self, memory_id: str, content: Optional[str] = None, metadata: Optional[dict] = None) -> bool:
        """Update a memory entry."""
        if memory_id not in self.memories:
            return False

        memory = self.memories[memory_id]

        if content:
            memory.content = content
            memory.embedding, cost = self._compute_embedding(content)
            print(f"[Memory] Updated embedding (cost: ${cost:.6f})")

        if metadata:
            memory.metadata.update(metadata)

        memory.accessed_at = datetime.now()
        self._save_memories()

        print(f"[Memory] Updated {memory_id}")
        return True

    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        if memory_id not in self.memories:
            return False

        del self.memories[memory_id]
        self._save_memories()

        print(f"[Memory] Deleted {memory_id}")
        return True

    # =========================================================================
    # Semantic Search (ME-03)
    # =========================================================================

    def semantic_search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> list[tuple[MemoryEntry, float]]:
        """
        Search memories by semantic similarity (ME-03).

        Returns list of (memory, similarity_score) tuples.
        """
        if not self.memories:
            return []

        query_embedding, _ = self._compute_embedding(query)
        results = []

        for memory in self.memories.values():
            if memory.embedding:
                similarity = self._cosine_similarity(query_embedding, memory.embedding)
                if similarity >= threshold:
                    results.append((memory, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # =========================================================================
    # Automatic Fact Extraction (ME-06)
    # =========================================================================

    def extract_facts(self, text: str) -> list[str]:
        """
        Extract facts from text for memory storage (ME-06).

        In production, would use NLP/LLM for extraction.
        """
        facts = []

        # Simple pattern-based extraction (demo)
        patterns = [
            r"(?:is|are|was|were)\s+(.+?)(?:\.|$)",
            r"(?:named?|called)\s+([A-Z][a-z]+)",
            r"(\d+)\s+(?:items?|records?|users?)",
        ]

        import re
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            facts.extend(matches)

        # Store extracted facts
        for fact in facts[:5]:  # Limit to 5 facts
            if len(fact) > 10:  # Only meaningful facts
                self.create(
                    content=fact.strip(),
                    source="extracted",
                    metadata={"source_text": text[:100]},
                )

        print(f"[Memory] Extracted {len(facts)} facts")
        return facts

    # =========================================================================
    # TTL and Cleanup (ME-07)
    # =========================================================================

    def cleanup_expired(self) -> int:
        """
        Remove expired memories based on TTL (ME-07).

        Returns count of removed memories.
        """
        expired = []
        now = datetime.now()

        for memory_id, memory in self.memories.items():
            if memory.ttl_hours is not None:
                expiry = memory.created_at + timedelta(hours=memory.ttl_hours)
                if now > expiry:
                    expired.append(memory_id)

        for memory_id in expired:
            del self.memories[memory_id]

        if expired:
            self._save_memories()
            print(f"[Memory] Cleaned up {len(expired)} expired memories")

        return len(expired)

    def cleanup_by_access(self, max_age_days: int = 30, min_access_count: int = 0) -> int:
        """
        Remove memories not accessed recently (ME-07).

        Returns count of removed memories.
        """
        to_remove = []
        cutoff = datetime.now() - timedelta(days=max_age_days)

        for memory_id, memory in self.memories.items():
            if memory.accessed_at < cutoff and memory.access_count <= min_access_count:
                to_remove.append(memory_id)

        for memory_id in to_remove:
            del self.memories[memory_id]

        if to_remove:
            self._save_memories()
            print(f"[Memory] Cleaned up {len(to_remove)} stale memories")

        return len(to_remove)

    # =========================================================================
    # Cost Tracking (ME-08)
    # =========================================================================

    def get_embedding_cost(self) -> float:
        """Get total embedding cost (ME-08)."""
        return self.total_embedding_cost

    def reset_cost_tracking(self):
        """Reset cost tracking."""
        self.total_embedding_cost = 0.0

    def get_stats(self) -> dict:
        """Get memory store statistics."""
        return {
            "total_memories": len(self.memories),
            "total_embedding_cost": self.total_embedding_cost,
            "sources": {
                source: len([m for m in self.memories.values() if m.source == source])
                for source in set(m.source for m in self.memories.values())
            },
            "with_ttl": len([m for m in self.memories.values() if m.ttl_hours is not None]),
        }


# =============================================================================
# Agent Autonomous Memory (ME-05)
# =============================================================================

class AutonomousMemoryAgent:
    """
    Demonstrates agent autonomous memory management (ME-05).

    Agent can decide what to remember and forget.
    """

    def __init__(self, memory_store: EnhancedMemoryStore):
        self.memory = memory_store
        self.importance_threshold = 0.5

    def should_remember(self, content: str) -> bool:
        """Agent decides if content is worth remembering."""
        # Simple heuristics (in production, would use LLM)
        importance = 0.0

        # Length indicates detail
        if len(content) > 50:
            importance += 0.3

        # Contains key phrases
        key_phrases = ["important", "remember", "note", "key", "critical"]
        if any(phrase in content.lower() for phrase in key_phrases):
            importance += 0.3

        # Contains numbers (often factual)
        if any(c.isdigit() for c in content):
            importance += 0.2

        return importance >= self.importance_threshold

    def process_and_remember(self, content: str) -> Optional[str]:
        """Process content and autonomously decide to remember."""
        if self.should_remember(content):
            memory_id = self.memory.create(
                content=content,
                source="autonomous",
                metadata={"agent_decision": "important"},
            )
            print(f"[Autonomous] Decided to remember: {content[:50]}...")
            return memory_id
        else:
            print(f"[Autonomous] Decided NOT to remember: {content[:30]}...")
            return None


def main():
    print("=" * 60)
    print("Memory: Basic Memory Test")
    print("=" * 60)
    print("""
This example demonstrates CrewAI's memory feature.
When memory=True, agents can remember information across tasks.

Memory Types in CrewAI:
1. Short-term Memory: Within the same crew execution
2. Long-term Memory: Persists across executions
3. Entity Memory: Remembers information about entities

LangGraph Comparison:
- CrewAI: memory=True (automatic)
- LangGraph: Explicit state management in graph
""")

    # ==========================================================================
    # Agent with memory enabled
    # ==========================================================================
    assistant = Agent(
        role="Research Assistant",
        goal="Help users by researching and remembering information",
        backstory="""You are a helpful research assistant with excellent
        memory. You remember details from previous interactions and use
        them to provide better assistance.""",
        verbose=True,
        memory=True,  # Enable agent-level memory
    )

    # ==========================================================================
    # Tasks that build on each other
    # ==========================================================================

    # Task 1: Learn information
    learn_task = Task(
        description="""Learn and memorize the following information about Project Alpha:
        - Project Code: ALPHA-2024
        - Budget: $2.5 million
        - Team Lead: Dr. Sarah Chen
        - Start Date: January 2024
        - Primary Goal: Develop an AI-powered analytics platform

        Confirm that you have memorized this information.""",
        expected_output="Confirmation that the information has been memorized",
        agent=assistant,
    )

    # Task 2: Recall information (tests short-term memory)
    recall_task = Task(
        description="""Without being given the information again, answer
        these questions about Project Alpha:
        1. What is the project code?
        2. Who is the team lead?
        3. What is the budget?

        This tests your short-term memory from the previous task.""",
        expected_output="Answers to all three questions from memory",
        agent=assistant,
    )

    # Task 3: Apply remembered information
    apply_task = Task(
        description="""Using your memory of Project Alpha, create a brief
        project summary email that could be sent to stakeholders.
        Include all the key details you remember.""",
        expected_output="A professional email summarizing Project Alpha",
        agent=assistant,
    )

    # ==========================================================================
    # Crew with memory enabled
    # ==========================================================================
    crew = Crew(
        agents=[assistant],
        tasks=[learn_task, recall_task, apply_task],
        verbose=True,
        memory=True,  # Enable crew-level memory
        # embedder configuration can be customized:
        # embedder={
        #     "provider": "openai",
        #     "config": {"model": "text-embedding-3-small"}
        # }
    )

    print("\n" + "=" * 60)
    print("Executing Memory Test (3 Tasks)")
    print("=" * 60)
    print("""
Task Flow:
1. Learn: Agent memorizes project information
2. Recall: Agent answers questions from memory
3. Apply: Agent uses memory to create a document

Watch for memory usage in the agent's reasoning.
""")

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    print(result)

    # Show task outputs for detailed analysis
    if result.tasks_output:
        print("\n" + "=" * 60)
        print("Individual Task Outputs:")
        print("=" * 60)
        for i, task_output in enumerate(result.tasks_output):
            print(f"\nTask {i+1}:")
            print("-" * 40)
            print(task_output.raw[:500] if len(task_output.raw) > 500 else task_output.raw)


def test_multi_agent_memory():
    """Test memory sharing between multiple agents."""
    print("\n" + "=" * 60)
    print("Multi-Agent Memory Test")
    print("=" * 60)

    # Two agents that should share memory
    agent_a = Agent(
        role="Information Gatherer",
        goal="Gather and store information",
        backstory="You collect information and pass it to colleagues.",
        verbose=True,
        memory=True,
    )

    agent_b = Agent(
        role="Information Processor",
        goal="Process information gathered by others",
        backstory="You work with information collected by your colleagues.",
        verbose=True,
        memory=True,
    )

    task_gather = Task(
        description="""Record the following facts:
        - The secret code is 'OMEGA-777'
        - The meeting is at 3 PM
        - The contact person is John""",
        expected_output="Confirmation of recorded facts",
        agent=agent_a,
    )

    task_process = Task(
        description="""Based on the information gathered by your colleague,
        what is the secret code and when is the meeting?
        (Note: This tests if memory is shared between agents)""",
        expected_output="The secret code and meeting time",
        agent=agent_b,
    )

    crew = Crew(
        agents=[agent_a, agent_b],
        tasks=[task_gather, task_process],
        verbose=True,
        memory=True,
    )

    result = crew.kickoff()
    print("\nMulti-Agent Memory Result:")
    print(result)


# =============================================================================
# Enhanced Memory Demonstrations
# =============================================================================

def demo_semantic_search():
    """Demonstrate semantic search (ME-03)."""
    print("\n" + "=" * 60)
    print("Demo: Semantic Search (ME-03)")
    print("=" * 60)

    store = EnhancedMemoryStore(Path("./db/demo_memory"))

    # Add some memories
    store.create("Python is a programming language", metadata={"topic": "programming"})
    store.create("Machine learning uses algorithms to learn from data", metadata={"topic": "ml"})
    store.create("CrewAI is a framework for building AI agents", metadata={"topic": "ai"})
    store.create("Tokyo is the capital of Japan", metadata={"topic": "geography"})

    # Search
    print("\nSearching for 'AI frameworks'...")
    results = store.semantic_search("AI frameworks", top_k=3)
    for memory, score in results:
        print(f"  [{score:.3f}] {memory.content}")


def demo_memory_crud():
    """Demonstrate memory CRUD operations (ME-04)."""
    print("\n" + "=" * 60)
    print("Demo: Memory CRUD (ME-04)")
    print("=" * 60)

    store = EnhancedMemoryStore(Path("./db/crud_memory"))

    # Create
    print("\n--- Create ---")
    mid = store.create("Important project deadline is next Friday", ttl_hours=48)

    # Read
    print("\n--- Read ---")
    memory = store.read(mid)
    if memory:
        print(f"Content: {memory.content}")
        print(f"Access count: {memory.access_count}")

    # Update
    print("\n--- Update ---")
    store.update(mid, metadata={"priority": "high"})
    memory = store.read(mid)
    if memory:
        print(f"Updated metadata: {memory.metadata}")

    # Delete
    print("\n--- Delete ---")
    store.delete(mid)
    print(f"After delete: {store.read(mid)}")


def demo_autonomous_memory():
    """Demonstrate autonomous memory management (ME-05)."""
    print("\n" + "=" * 60)
    print("Demo: Autonomous Memory (ME-05)")
    print("=" * 60)

    store = EnhancedMemoryStore(Path("./db/auto_memory"))
    agent = AutonomousMemoryAgent(store)

    # Test various content
    contents = [
        "The meeting time is 3pm - important!",  # Should remember
        "ok",  # Should not remember
        "Key finding: 42% increase in sales this quarter",  # Should remember
        "hello",  # Should not remember
        "Remember to follow up with the client about the 1000 item order",  # Should remember
    ]

    for content in contents:
        agent.process_and_remember(content)


def demo_fact_extraction():
    """Demonstrate automatic fact extraction (ME-06)."""
    print("\n" + "=" * 60)
    print("Demo: Fact Extraction (ME-06)")
    print("=" * 60)

    store = EnhancedMemoryStore(Path("./db/fact_memory"))

    text = """
    The company is called TechCorp. It was founded in 2010.
    The CEO is named Sarah Johnson. There are 500 employees.
    The main product is an AI platform that processes 1 million records daily.
    """

    print(f"Extracting facts from:\n{text[:100]}...")
    facts = store.extract_facts(text)
    print(f"\nExtracted facts: {facts}")


def demo_ttl_cleanup():
    """Demonstrate TTL and cleanup (ME-07)."""
    print("\n" + "=" * 60)
    print("Demo: TTL and Cleanup (ME-07)")
    print("=" * 60)

    store = EnhancedMemoryStore(Path("./db/ttl_memory"))

    # Add memories with different TTLs
    store.create("Short-lived memory", ttl_hours=0)  # Immediately expired
    store.create("Normal memory", ttl_hours=24)
    store.create("Long-lived memory", ttl_hours=720)  # 30 days

    print(f"\nBefore cleanup: {len(store.memories)} memories")
    expired_count = store.cleanup_expired()
    print(f"After cleanup: {len(store.memories)} memories ({expired_count} removed)")


def demo_embedding_cost():
    """Demonstrate embedding cost tracking (ME-08)."""
    print("\n" + "=" * 60)
    print("Demo: Embedding Cost Tracking (ME-08)")
    print("=" * 60)

    store = EnhancedMemoryStore(Path("./db/cost_memory"))
    store.reset_cost_tracking()

    # Create several memories
    texts = [
        "Short text",
        "A longer piece of text that contains more tokens and will cost more to embed",
        "Another text with some content about AI and machine learning concepts",
    ]

    for text in texts:
        store.create(text)

    print(f"\nTotal embedding cost: ${store.get_embedding_cost():.6f}")
    print(f"Stats: {store.get_stats()}")


if __name__ == "__main__":
    main()

    # Uncomment to run additional demos:
    # test_multi_agent_memory()
    # demo_semantic_search()
    # demo_memory_crud()
    # demo_autonomous_memory()
    # demo_fact_extraction()
    # demo_ttl_cleanup()
    # demo_embedding_cost()
