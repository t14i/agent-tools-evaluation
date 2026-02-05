# CrewAI Sample

Verification repository for CrewAI's production readiness, comparing with LangGraph.

## Setup

```bash
# Install dependencies
uv sync

# Copy .env.example to .env and set your API key
cp .env.example .env

# Run scripts
uv run --env-file .env python 01_quickstart.py
```

## Files

| File | Description |
|------|-------------|
| `01_quickstart.py` | Quick Start - Minimal Agent/Task/Crew |
| `02_tool_definition.py` | Tool definition methods comparison |
| `03_tool_execution.py` | Tool execution, caching |
| `04_tool_error_handling.py` | Error handling, retry pattern |
| `05_hitl_task_input.py` | HITL basics - human_input=True |
| `06_hitl_flow_feedback.py` | HITL with Flow - feedback loop |
| `07_durable_basic.py` | Durable execution basics (@persist) |
| `08_durable_resume.py` | Durable execution resume test |
| `09_collaboration_delegation.py` | Multi-agent delegation |
| `10_collaboration_hierarchical.py` | Hierarchical process |
| `11_memory_basic.py` | Memory basics - memory=True |
| `12_memory_longterm.py` | Long-term memory persistence |
| `13_production_concerns.py` | Production concerns verification |
| `REPORT.md` | Detailed verification report |
| `REPORT_ja.md` | Japanese version of the report |

## Key Findings

### Tool Calling
- `@tool` decorator is simple and clean
- `@tool` does NOT support `args_schema` or `cache=True` (use `BaseTool` for these)
- Error propagation to agent is automatic
- `max_retry_limit` controls retry behavior

### HITL
- `human_input=True` on Task is simple but limited
- Flow-based HITL with `@start/@listen/@router` is more flexible
- Less granular control than LangGraph's `interrupt()`
- Production requires additional infrastructure:
  - Custom UI integration
  - Timeout handling
  - Notification system

### Durable Execution
- `@persist(persistence=SQLiteFlowPersistence(...))` for Flow-based workflows
- Resume with `kickoff(inputs={'id': state_id})` - triggers automatic state restoration
- Each method should check state to skip completed work
- Less explicit control than LangGraph's Checkpointer

### Role-Based Collaboration (CrewAI-specific)
- `allow_delegation=True` enables automatic delegation
- `Process.hierarchical` auto-generates manager agent
- No equivalent in LangGraph (requires manual implementation)

### Memory
- `memory=True` enables short-term and long-term memory
- Long-term memory persists across sessions (verified)
- Entity memory tracks information about entities
- Less explicit than LangGraph's state management

## Key Gotchas

1. `@tool` does NOT support `args_schema` or `cache=True` parameters
2. `@listen` requires method reference, not string: `@listen(step1)` not `@listen("step1")`
3. `Process.hierarchical` requires `manager_llm` or `manager_agent`
4. `@persist` requires `kickoff(inputs={'id': ...})` to resume - state is auto-restored with same ID

See [REPORT.md](./REPORT.md) for details.
