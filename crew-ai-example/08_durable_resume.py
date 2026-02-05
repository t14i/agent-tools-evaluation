"""
08_durable_resume.py - Durable Execution Resume Test

Purpose: Verify resume functionality using @persist decorator
- Automatic state persistence with @persist
- Resume from interruption point using kickoff(inputs={'id': state_id})
- State consistency across restarts

LangGraph Comparison:
- LangGraph: Checkpointer + thread_id
- CrewAI: @persist decorator + kickoff(inputs={'id': state_id})

Usage:
1. First run: python 08_durable_resume.py --start
2. Press Ctrl+C to interrupt
3. Resume: python 08_durable_resume.py --resume <state_id>
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from crewai.flow.flow import Flow, listen, start
from crewai.flow.persistence import persist, SQLiteFlowPersistence
from pydantic import BaseModel, Field


# Ensure db directory exists
DB_PATH = Path("./db")
DB_PATH.mkdir(exist_ok=True)


class LongRunningState(BaseModel):
    """State for a long-running workflow that may need to be resumed."""

    id: str = ""  # UUID managed by @persist
    flow_id: str = ""
    current_phase: int = 0
    total_phases: int = 5
    results: list = Field(default_factory=list)
    is_complete: bool = False
    started_at: str = ""


# Use SQLite for persistence across process restarts
sqlite_persistence = SQLiteFlowPersistence(db_path="./db/flow_state.db")


@persist(persistence=sqlite_persistence, verbose=True)
class ResumableWorkflow(Flow[LongRunningState]):
    """
    Workflow demonstrating @persist for automatic state persistence.

    Key: Use kickoff(inputs={'id': state_id}) to resume from persisted state.
    """

    @start()
    def phase_0_initialize(self):
        """Initialize the workflow."""
        if self.state.current_phase > 0:
            print(f"\n[Resume] Detected state at phase {self.state.current_phase}, skipping init...")
            return

        print("\n[Phase 0] Initializing workflow...")
        self.state.flow_id = f"flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state.started_at = datetime.now().isoformat()
        self.state.results.append({"phase": 0, "action": "initialize"})
        print(f"[Phase 0] State ID: {self.state.id}")

    @listen(phase_0_initialize)
    def phase_1_data_collection(self):
        """Phase 1: Data collection."""
        if self.state.current_phase >= 1:
            print("[Phase 1] Already completed, skipping...")
            return

        print("\n[Phase 1] Collecting data (1 second)...")
        self.state.current_phase = 1
        time.sleep(1)
        self.state.results.append({"phase": 1, "action": "data_collection"})
        print("[Phase 1] Complete.")

    @listen(phase_1_data_collection)
    def phase_2_validation(self):
        """Phase 2: Validation."""
        if self.state.current_phase >= 2:
            print("[Phase 2] Already completed, skipping...")
            return

        print("\n[Phase 2] Validating (1 second)...")
        self.state.current_phase = 2
        time.sleep(1)
        self.state.results.append({"phase": 2, "action": "validation"})
        print("[Phase 2] Complete.")

    @listen(phase_2_validation)
    def phase_3_processing(self):
        """Phase 3: Processing (interruptible)."""
        if self.state.current_phase >= 3:
            print("[Phase 3] Already completed, skipping...")
            return

        print("\n[Phase 3] Processing (3 seconds, interruptible)...")
        self.state.current_phase = 3

        for i in range(3):
            print(f"[Phase 3] Batch {i+1}/3...")
            time.sleep(1)

        self.state.results.append({"phase": 3, "action": "processing"})
        print("[Phase 3] Complete.")

    @listen(phase_3_processing)
    def phase_4_aggregation(self):
        """Phase 4: Aggregation."""
        if self.state.current_phase >= 4:
            print("[Phase 4] Already completed, skipping...")
            return

        print("\n[Phase 4] Aggregating (1 second)...")
        self.state.current_phase = 4
        time.sleep(1)
        self.state.results.append({"phase": 4, "action": "aggregation"})
        print("[Phase 4] Complete.")

    @listen(phase_4_aggregation)
    def phase_5_finalize(self):
        """Phase 5: Finalization."""
        if self.state.is_complete:
            print("[Phase 5] Already completed, skipping...")
            return self._report()

        print("\n[Phase 5] Finalizing...")
        self.state.current_phase = 5
        self.state.is_complete = True
        self.state.results.append({"phase": 5, "action": "finalize"})
        print("[Phase 5] Complete!")
        return self._report()

    def _report(self):
        return f"""
Workflow Report
===============
State ID: {self.state.id}
Phases: {self.state.current_phase}/{self.state.total_phases}
Status: {'COMPLETE' if self.state.is_complete else 'INCOMPLETE'}
"""


def main():
    parser = argparse.ArgumentParser(description="Resumable Workflow with @persist")
    parser.add_argument("--start", action="store_true", help="Start new workflow")
    parser.add_argument("--resume", type=str, help="Resume with state ID")
    args = parser.parse_args()

    print("=" * 60)
    print("Durable Execution: @persist Resume Test")
    print("=" * 60)

    if args.resume:
        # Verify state exists
        state = sqlite_persistence.load_state(args.resume)
        if not state:
            print(f"Error: No state found for ID {args.resume}")
            sys.exit(1)

        print(f"\nResuming from phase {state.get('current_phase')}...")

        flow = ResumableWorkflow()
        try:
            # KEY: Pass ID via kickoff inputs to restore state
            result = flow.kickoff(inputs={'id': args.resume})
            print("\n" + "=" * 60)
            print(result)
        except KeyboardInterrupt:
            print(f"\n\nInterrupted! Resume with: --resume {flow.state.id}")

    elif args.start:
        print("\nStarting new workflow (Ctrl+C to interrupt)...\n")

        flow = ResumableWorkflow()
        try:
            result = flow.kickoff()
            print("\n" + "=" * 60)
            print(result)
        except KeyboardInterrupt:
            print(f"\n\nInterrupted at phase {flow.state.current_phase}!")
            print(f"State ID: {flow.state.id}")
            print(f"Resume with: python 08_durable_resume.py --resume {flow.state.id}")

    else:
        print("""
Usage:
  python 08_durable_resume.py --start          Start new workflow
  python 08_durable_resume.py --resume <id>    Resume from state ID

How @persist works:
1. State saved to SQLite after each method
2. Pass same ID via kickoff(inputs={'id': ...}) to restore
3. Check current_phase in each method to skip completed work

Key Discovery:
  kickoff(inputs={'id': state_id}) triggers automatic state restoration!
""")


if __name__ == "__main__":
    main()
