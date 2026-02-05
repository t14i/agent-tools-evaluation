# Temporal Durable Execution Evaluation

This project evaluates Temporal for production readiness based on the Durable Execution Evaluation Criteria.

## Setup

```bash
# Install dependencies
uv sync

# Start Temporal dev server
temporal server start-dev

# Run verification scripts
uv run python 01_ex_execution_semantics.py
```

## Test Environment

- Python: 3.13
- Temporal SDK: 1.9.x
- Temporal Server: Dev mode (localhost:7233)
