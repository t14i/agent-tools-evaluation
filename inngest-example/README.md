# Inngest Durable Execution Evaluation

This project evaluates Inngest (Python SDK) for production readiness based on the Durable Execution Evaluation Criteria.

## Architecture Differences from Temporal

| Aspect | Temporal | Inngest |
|--------|----------|---------|
| Execution Model | Event Sourcing + Replay | Journal-based + Step memoization |
| Server | gRPC (localhost:7233) | HTTP (localhost:8288) |
| Function Definition | @workflow.defn + Worker | @inngest.create_function + HTTP serve |
| Step Definition | @activity.defn | step.run() |
| Determinism Constraint | Strict (entire code) | Relaxed (only within steps) |
| Deployment | Worker process | Serverless HTTP endpoint |

## Setup

```bash
# Install dependencies
uv sync

# Start Inngest Dev Server (requires Node.js)
npx inngest-cli@latest dev

# Start the application server
uv run uvicorn serve:app --reload --port 8000

# Run verification scripts (in another terminal)
uv run python 01_ex_execution_semantics.py
```

## Test Environment

- Python: 3.13
- Inngest SDK: 0.5.x
- Inngest Dev Server: latest (localhost:8288)
- App Server: FastAPI + Uvicorn (localhost:8000)

## How It Works

1. **Dev Server** (`npx inngest-cli@latest dev`): Runs the Inngest orchestration engine
2. **App Server** (`uvicorn serve:app`): Exposes Inngest functions via HTTP
3. **Verification Scripts**: Trigger events and validate behavior

## File Structure

```
inngest-example/
├── serve.py                      # FastAPI server to serve Inngest functions
├── common.py                     # Shared utilities
├── 01_ex_execution_semantics.py  # EX category verification
├── 02_rt_retry_timeout.py        # RT category verification
├── 03_wf_workflow_primitives.py  # WF category verification
├── 04_sg_signals_events.py       # SG category verification
├── 05_vr_versioning.py           # VR category verification
├── 06_cp_compensation.py         # CP category verification
├── 07_pf_performance.py          # PF category verification
├── 08_op_operations.py           # OP category verification
├── 09_ob_observability.py        # OB category verification
├── 10_dx_developer_experience.py # DX category verification
├── 11_ai_agent_integration.py    # AI category verification
├── REPORT.md                     # Evaluation report
└── pyproject.toml
```
