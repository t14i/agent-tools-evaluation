"""
18_determinism_evidence.py - Determinism Evidence (DR-02, DR-03)

Purpose: Verify evidence references and non-determinism isolation
- DR-02: Reference evidence - what each decision was based on
- DR-03: Non-determinism isolation - isolate and fix LLM/external I/O
- Input snapshot preservation
- LLM-disabled execution mode

LangGraph Comparison:
- Both require custom implementation for evidence tracking
- CrewAI has no built-in evidence collection mechanism
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field


# =============================================================================
# Evidence Collection (DR-02)
# =============================================================================

class EvidenceType(str, Enum):
    """Types of evidence that can be collected."""
    INPUT_DATA = "input_data"
    EXTERNAL_API = "external_api"
    DATABASE_QUERY = "database_query"
    LLM_RESPONSE = "llm_response"
    COMPUTED_VALUE = "computed_value"
    RULE_MATCH = "rule_match"
    USER_INPUT = "user_input"
    CONFIG = "config"


@dataclass
class Evidence:
    """A single piece of evidence supporting a decision."""

    id: str
    evidence_type: EvidenceType
    source: str  # Where the evidence came from
    value: Any  # The actual evidence value
    timestamp: str
    context: dict = field(default_factory=dict)
    hash: str = ""  # For verification

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.evidence_type.value,
            "source": self.source,
            "value": self.value,
            "timestamp": self.timestamp,
            "context": self.context,
            "hash": self.hash,
        }


@dataclass
class Decision:
    """A decision made during execution with its supporting evidence."""

    id: str
    description: str
    outcome: str
    evidence_ids: list[str]
    timestamp: str
    confidence: Optional[float] = None
    rationale: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "outcome": self.outcome,
            "evidence_ids": self.evidence_ids,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


class EvidenceCollector:
    """
    Collects and tracks evidence for decisions (DR-02).

    Every decision should be traceable to the evidence that supported it.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./db/evidence")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.evidence: dict[str, Evidence] = {}
        self.decisions: dict[str, Decision] = {}
        self.execution_id: str = ""

    def start_execution(self, execution_id: Optional[str] = None):
        """Start a new execution context."""
        self.execution_id = execution_id or f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.evidence = {}
        self.decisions = {}
        print(f"[EvidenceCollector] Started execution: {self.execution_id}")

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID."""
        count = len(self.evidence) + len(self.decisions)
        return f"{prefix}_{self.execution_id}_{count:04d}"

    def _compute_hash(self, value: Any) -> str:
        """Compute hash of value for verification."""
        import hashlib
        content = json.dumps(value, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def record_evidence(
        self,
        evidence_type: EvidenceType,
        source: str,
        value: Any,
        context: Optional[dict] = None,
    ) -> str:
        """
        Record a piece of evidence.

        Returns the evidence ID.
        """
        evidence_id = self._generate_id("ev")

        evidence = Evidence(
            id=evidence_id,
            evidence_type=evidence_type,
            source=source,
            value=value,
            timestamp=datetime.now().isoformat(),
            context=context or {},
            hash=self._compute_hash(value),
        )

        self.evidence[evidence_id] = evidence
        print(f"[Evidence] Recorded {evidence_type.value}: {source} -> {evidence_id}")
        return evidence_id

    def record_decision(
        self,
        description: str,
        outcome: str,
        evidence_ids: list[str],
        confidence: Optional[float] = None,
        rationale: Optional[str] = None,
    ) -> str:
        """
        Record a decision with its supporting evidence.

        Returns the decision ID.
        """
        decision_id = self._generate_id("dec")

        # Validate evidence IDs
        for eid in evidence_ids:
            if eid not in self.evidence:
                print(f"[Warning] Evidence {eid} not found")

        decision = Decision(
            id=decision_id,
            description=description,
            outcome=outcome,
            evidence_ids=evidence_ids,
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            rationale=rationale,
        )

        self.decisions[decision_id] = decision
        print(f"[Decision] Recorded: {description} -> {outcome}")
        return decision_id

    def get_decision_evidence(self, decision_id: str) -> list[Evidence]:
        """Get all evidence for a decision."""
        if decision_id not in self.decisions:
            return []

        decision = self.decisions[decision_id]
        return [
            self.evidence[eid]
            for eid in decision.evidence_ids
            if eid in self.evidence
        ]

    def save_to_file(self):
        """Save evidence and decisions to file."""
        output_file = self.storage_path / f"{self.execution_id}.json"

        data = {
            "execution_id": self.execution_id,
            "timestamp": datetime.now().isoformat(),
            "evidence": {eid: e.to_dict() for eid, e in self.evidence.items()},
            "decisions": {did: d.to_dict() for did, d in self.decisions.items()},
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"[EvidenceCollector] Saved to {output_file}")

    def generate_report(self) -> str:
        """Generate a human-readable evidence report."""
        lines = [
            f"Evidence Report: {self.execution_id}",
            "=" * 60,
            "",
            f"Total Evidence Items: {len(self.evidence)}",
            f"Total Decisions: {len(self.decisions)}",
            "",
            "Decisions and Supporting Evidence:",
            "-" * 40,
        ]

        for did, decision in self.decisions.items():
            lines.append(f"\n[{did}] {decision.description}")
            lines.append(f"  Outcome: {decision.outcome}")
            if decision.confidence:
                lines.append(f"  Confidence: {decision.confidence:.2%}")
            if decision.rationale:
                lines.append(f"  Rationale: {decision.rationale}")
            lines.append("  Evidence:")
            for eid in decision.evidence_ids:
                if eid in self.evidence:
                    ev = self.evidence[eid]
                    lines.append(f"    - [{ev.evidence_type.value}] {ev.source}")
                    lines.append(f"      Value: {str(ev.value)[:100]}...")

        return "\n".join(lines)


# =============================================================================
# Input Snapshot (DR-03)
# =============================================================================

class InputSnapshot:
    """
    Captures and stores input snapshots for reproducibility.

    Allows replaying with exact same inputs even if external
    sources have changed.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./db/snapshots")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.snapshot_id: str = ""
        self.data: dict = {}

    def create(self, snapshot_id: Optional[str] = None) -> str:
        """Create a new snapshot."""
        self.snapshot_id = snapshot_id or f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.data = {
            "id": self.snapshot_id,
            "created_at": datetime.now().isoformat(),
            "inputs": {},
        }
        return self.snapshot_id

    def capture(self, name: str, value: Any, source: str = "unknown"):
        """Capture an input value."""
        self.data["inputs"][name] = {
            "value": value,
            "source": source,
            "captured_at": datetime.now().isoformat(),
        }
        print(f"[Snapshot] Captured: {name} from {source}")

    def save(self):
        """Save snapshot to file."""
        output_file = self.storage_path / f"{self.snapshot_id}.json"
        with open(output_file, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        print(f"[Snapshot] Saved to {output_file}")

    def load(self, snapshot_id: str) -> dict:
        """Load a previously saved snapshot."""
        input_file = self.storage_path / f"{snapshot_id}.json"
        if not input_file.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")

        with open(input_file, "r") as f:
            self.data = json.load(f)
            self.snapshot_id = snapshot_id

        print(f"[Snapshot] Loaded: {snapshot_id}")
        return self.data

    def get(self, name: str) -> Any:
        """Get a captured input value."""
        if name in self.data.get("inputs", {}):
            return self.data["inputs"][name]["value"]
        return None


# =============================================================================
# Deterministic Mode (DR-03)
# =============================================================================

class DeterministicMode:
    """
    Context manager for deterministic execution mode.

    In deterministic mode:
    - LLM calls are disabled or return mocked values
    - External I/O uses snapshots
    - Random values use fixed seeds
    """

    def __init__(
        self,
        enable_llm: bool = False,
        snapshot: Optional[InputSnapshot] = None,
        random_seed: int = 42,
    ):
        self.enable_llm = enable_llm
        self.snapshot = snapshot
        self.random_seed = random_seed
        self._original_random_state = None

    def __enter__(self):
        """Enter deterministic mode."""
        import random
        self._original_random_state = random.getstate()
        random.seed(self.random_seed)

        if not self.enable_llm:
            print("[DeterministicMode] LLM calls DISABLED")

        print(f"[DeterministicMode] Random seed: {self.random_seed}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit deterministic mode."""
        import random
        if self._original_random_state:
            random.setstate(self._original_random_state)

        print("[DeterministicMode] Exited")
        return False

    def call_llm(self, prompt: str) -> str:
        """
        Call LLM if enabled, otherwise return placeholder.

        In production, this would integrate with actual LLM.
        """
        if self.enable_llm:
            # Would call actual LLM here
            return f"[LLM Response to: {prompt[:50]}...]"
        else:
            return f"[MOCK LLM - Deterministic mode - prompt: {prompt[:50]}...]"

    def get_external_input(self, name: str) -> Any:
        """Get external input from snapshot if available."""
        if self.snapshot:
            value = self.snapshot.get(name)
            if value is not None:
                print(f"[DeterministicMode] Using snapshot value for: {name}")
                return value

        print(f"[DeterministicMode] No snapshot value for: {name}")
        return None


# =============================================================================
# Evidence-Tracked Workflow State
# =============================================================================

class EvidenceTrackedState(BaseModel):
    """State for evidence-tracked workflow."""

    execution_id: str = ""
    snapshot_id: str = ""
    deterministic_mode: bool = False
    decisions_made: list[str] = []
    current_step: str = "init"


# =============================================================================
# Evidence-Tracked Workflow
# =============================================================================

class EvidenceTrackedWorkflow(Flow[EvidenceTrackedState]):
    """
    Workflow demonstrating evidence collection and deterministic execution.

    Every decision is tracked with its supporting evidence.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evidence_collector = EvidenceCollector()
        self.snapshot = InputSnapshot()

    @start()
    def initialize(self):
        """Initialize workflow with evidence tracking."""
        print("\n[Initialize] Setting up evidence tracking...")

        self.state.execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.evidence_collector.start_execution(self.state.execution_id)

        # Record initialization as evidence
        self.evidence_collector.record_evidence(
            evidence_type=EvidenceType.CONFIG,
            source="workflow_init",
            value={"deterministic_mode": self.state.deterministic_mode},
        )

    @listen(initialize)
    def gather_inputs(self):
        """Gather and snapshot inputs."""
        print("\n[Gather] Collecting inputs...")

        # Create snapshot
        self.state.snapshot_id = self.snapshot.create()

        # Simulate gathering external data
        external_data = {
            "market_data": {"price": 100.50, "volume": 1000},
            "user_preferences": {"risk_level": "medium"},
            "system_config": {"max_allocation": 0.1},
        }

        for name, value in external_data.items():
            self.snapshot.capture(name, value, source="external_api")

            # Record as evidence
            self.evidence_collector.record_evidence(
                evidence_type=EvidenceType.EXTERNAL_API,
                source=f"api.{name}",
                value=value,
            )

        self.snapshot.save()

    @listen(gather_inputs)
    def make_decisions(self):
        """Make decisions with evidence tracking."""
        print("\n[Decide] Making decisions based on evidence...")

        # Get evidence IDs for inputs
        evidence_ids = list(self.evidence_collector.evidence.keys())

        # Decision 1: Risk assessment
        market_data = self.snapshot.get("market_data")
        user_prefs = self.snapshot.get("user_preferences")

        risk_score = 0.5  # Simulated calculation
        if market_data and market_data.get("volume", 0) > 500:
            risk_score += 0.1
        if user_prefs and user_prefs.get("risk_level") == "medium":
            risk_score = min(risk_score, 0.7)

        # Record computed value as evidence
        risk_evidence_id = self.evidence_collector.record_evidence(
            evidence_type=EvidenceType.COMPUTED_VALUE,
            source="risk_calculator",
            value={"risk_score": risk_score},
            context={"inputs": ["market_data", "user_preferences"]},
        )

        # Record decision
        decision1_id = self.evidence_collector.record_decision(
            description="Risk Assessment",
            outcome=f"risk_score={risk_score}",
            evidence_ids=evidence_ids + [risk_evidence_id],
            confidence=0.85,
            rationale="Based on market volume and user risk preference",
        )
        self.state.decisions_made.append(decision1_id)

        # Decision 2: Allocation decision
        config = self.snapshot.get("system_config")
        max_alloc = config.get("max_allocation", 0.1) if config else 0.1

        allocation = max_alloc * (1 - risk_score)

        alloc_evidence_id = self.evidence_collector.record_evidence(
            evidence_type=EvidenceType.COMPUTED_VALUE,
            source="allocation_calculator",
            value={"allocation": allocation},
        )

        decision2_id = self.evidence_collector.record_decision(
            description="Allocation Decision",
            outcome=f"allocation={allocation:.4f}",
            evidence_ids=[risk_evidence_id, alloc_evidence_id],
            confidence=0.9,
            rationale="Allocation inversely proportional to risk",
        )
        self.state.decisions_made.append(decision2_id)

    @listen(make_decisions)
    def finalize(self):
        """Finalize and generate report."""
        print("\n[Finalize] Generating evidence report...")

        self.evidence_collector.save_to_file()
        report = self.evidence_collector.generate_report()

        self.state.current_step = "complete"
        return report


# =============================================================================
# Demonstrations
# =============================================================================

def demo_evidence_collection():
    """Demonstrate evidence collection."""
    print("=" * 60)
    print("Demo 1: Evidence Collection (DR-02)")
    print("=" * 60)

    collector = EvidenceCollector()
    collector.start_execution()

    # Collect various evidence
    ev1 = collector.record_evidence(
        evidence_type=EvidenceType.INPUT_DATA,
        source="user_request",
        value={"query": "Find best investment options"},
    )

    ev2 = collector.record_evidence(
        evidence_type=EvidenceType.EXTERNAL_API,
        source="market_api",
        value={"stocks": [{"symbol": "AAPL", "price": 150}]},
    )

    ev3 = collector.record_evidence(
        evidence_type=EvidenceType.LLM_RESPONSE,
        source="gpt-4",
        value={"recommendation": "Consider tech stocks"},
        context={"temperature": 0, "model": "gpt-4"},
    )

    # Make a decision based on evidence
    collector.record_decision(
        description="Investment Recommendation",
        outcome="Recommend AAPL with moderate allocation",
        evidence_ids=[ev1, ev2, ev3],
        confidence=0.75,
        rationale="User wants investment options, market data shows tech strength, LLM suggests tech",
    )

    print("\n" + collector.generate_report())


def demo_input_snapshot():
    """Demonstrate input snapshotting."""
    print("\n" + "=" * 60)
    print("Demo 2: Input Snapshot (DR-03)")
    print("=" * 60)

    # Create and populate snapshot
    snapshot = InputSnapshot()
    snapshot_id = snapshot.create()

    snapshot.capture("api_response", {"data": [1, 2, 3]}, "external_api")
    snapshot.capture("user_input", "process these items", "stdin")
    snapshot.capture("config", {"timeout": 30}, "config_file")

    snapshot.save()

    # Load and use snapshot
    print("\nLoading snapshot for replay...")
    snapshot2 = InputSnapshot()
    snapshot2.load(snapshot_id)

    print(f"\nRetrieved values:")
    print(f"  api_response: {snapshot2.get('api_response')}")
    print(f"  user_input: {snapshot2.get('user_input')}")
    print(f"  config: {snapshot2.get('config')}")


def demo_deterministic_mode():
    """Demonstrate deterministic execution mode."""
    print("\n" + "=" * 60)
    print("Demo 3: Deterministic Mode (DR-03)")
    print("=" * 60)

    # Create snapshot first
    snapshot = InputSnapshot()
    snapshot.create("demo_snapshot")
    snapshot.capture("random_input", 42, "test")
    snapshot.save()

    print("\nRunning in deterministic mode...")
    with DeterministicMode(enable_llm=False, snapshot=snapshot, random_seed=42) as dm:
        # Random values should be deterministic
        import random
        print(f"  Random value 1: {random.random():.6f}")
        print(f"  Random value 2: {random.random():.6f}")

        # LLM calls return mock
        response = dm.call_llm("Analyze this data")
        print(f"  LLM response: {response}")

        # External input uses snapshot
        value = dm.get_external_input("random_input")
        print(f"  External input: {value}")

    print("\nRunning again with same seed (should be identical)...")
    with DeterministicMode(enable_llm=False, snapshot=snapshot, random_seed=42) as dm:
        import random
        print(f"  Random value 1: {random.random():.6f}")
        print(f"  Random value 2: {random.random():.6f}")


def demo_evidence_workflow():
    """Demonstrate evidence-tracked workflow."""
    print("\n" + "=" * 60)
    print("Demo 4: Evidence-Tracked Workflow")
    print("=" * 60)

    flow = EvidenceTrackedWorkflow()
    result = flow.kickoff()
    print(result)


def demo_decision_audit():
    """Demonstrate auditing decisions and their evidence."""
    print("\n" + "=" * 60)
    print("Demo 5: Decision Audit Trail")
    print("=" * 60)

    collector = EvidenceCollector()
    collector.start_execution("audit_demo")

    # Simulate a complex decision process
    ev1 = collector.record_evidence(
        evidence_type=EvidenceType.RULE_MATCH,
        source="policy_engine",
        value={"rule": "max_transaction_limit", "matched": True},
    )

    ev2 = collector.record_evidence(
        evidence_type=EvidenceType.DATABASE_QUERY,
        source="user_db",
        value={"user_tier": "premium", "account_age_days": 365},
    )

    dec1 = collector.record_decision(
        description="Transaction Approval Check",
        outcome="APPROVED",
        evidence_ids=[ev1, ev2],
        confidence=0.95,
        rationale="User is premium tier with good standing, within limits",
    )

    # Show decision audit
    print("\nDecision Audit:")
    print("-" * 40)
    print(f"Decision ID: {dec1}")

    evidence_list = collector.get_decision_evidence(dec1)
    print("\nSupporting Evidence:")
    for ev in evidence_list:
        print(f"  [{ev.evidence_type.value}] {ev.source}")
        print(f"    Value: {ev.value}")
        print(f"    Hash: {ev.hash}")
        print()


def main():
    print("=" * 60)
    print("Determinism Evidence Verification (DR-02, DR-03)")
    print("=" * 60)
    print("""
This script verifies evidence references and non-determinism isolation.

Verification Items:
- DR-02: Evidence References
  - Track what each decision was based on
  - Link decisions to supporting evidence
  - Generate audit reports

- DR-03: Non-determinism Isolation
  - Capture input snapshots
  - Mock external I/O for replay
  - LLM-disabled execution mode
  - Fixed random seeds

Key Components:
- EvidenceCollector: Records and links evidence to decisions
- InputSnapshot: Captures external inputs for replay
- DeterministicMode: Context manager for controlled execution

LangGraph Comparison:
- Neither has built-in evidence collection
- Neither has built-in deterministic mode
- Both require custom implementation for audit trails
""")

    # Run all demos
    demo_evidence_collection()
    demo_input_snapshot()
    demo_deterministic_mode()
    demo_evidence_workflow()
    demo_decision_audit()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
