# OpenAI Agents SDK Evaluation

Production readiness evaluation of OpenAI Agents SDK (openai-agents-python v0.7.0).

## Setup

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Running Scripts

```bash
# Run individual verification scripts
uv run python 01_quickstart.py
uv run python 02_tool_definition.py
# ... etc
```

## Evaluation Categories

| Category | Scripts | Focus |
|----------|---------|-------|
| TC | 02-04 | Tool Calling |
| HI | 05-06 | Human-in-the-Loop |
| DU | 07-09 | Durable Execution |
| ME | 10-12 | Memory |
| MA | 13-14 | Multi-Agent |
| GV | 15-16 | Governance |
| DR | 17-18 | Determinism & Replay |
| CX | 19-20 | Connectors |
| OB | 21-23 | Observability |
| TE | 24-25 | Testing & Evaluation |

## Key SDK Features

- **Agents**: Core agent abstraction with tools and handoffs
- **Handoffs**: Native multi-agent delegation
- **Guardrails**: Input/output validation
- **Tracing**: Built-in observability
- **Sessions**: State persistence

See [REPORT.md](./REPORT.md) for detailed evaluation results.
