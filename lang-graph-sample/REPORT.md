# LangGraph Verification Report

## Overview

This report summarizes the findings from evaluating LangGraph for production readiness based on the Agent Framework Evaluation Criteria (NIST AI RMF, WEF AI Agents in Action, IMDA Model AI Governance, OTel GenAI Semantic Conventions).

## Test Environment

- Python: 3.13
- LangGraph: 0.2.x
- LangChain: 0.3.x
- langmem: 0.0.x

---

## Star Rating Criteria

| Stars | Label | Definition | Judgment Criteria |
|-------|-------|------------|-------------------|
| ⭐ | Not Supported | No feature or broken | No documentation, doesn't work, requires complete custom implementation |
| ⭐⭐ | Experimental | Works but major constraints, struggles even in PoC | Works but many pitfalls, lacking docs, unstable API |
| ⭐⭐⭐ | PoC Ready | Basic functionality OK, usable for demo but needs additional work for production | Main cases work, weak on edge cases, monitoring/logging custom |
| ⭐⭐⭐⭐ | Production Ready | Practical, can deploy to production with minor customization | Stable, well documented, production cases exist |
| ⭐⭐⭐⭐⭐ | Production Recommended | Can use as-is in production, best practices established | Large-scale production cases, mature ecosystem |

---

## Prerequisite Parameters

| Parameter | Value |
|-----------|-------|
| Autonomy | Approval Required |
| Authority | Restricted Write |
| Predictability | LLM Decision Involved |
| Context | Internal Data |

---

## Evaluation Summary by Category

### Coverage Summary (57 Items)

| Category | Items | Good (⭐⭐⭐+) | Not Good (⭐⭐-) | Notes |
|----------|-------|---------------|-----------------|-------|
| TC: Tool Calling | 5 | 4 | 1 | Tool definition excellent, controllable automation weak |
| HI: Human-in-the-Loop | 5 | 3 | 2 | interrupt() is solid, timeout/notification missing |
| DU: Durable Execution | 6 | 4 | 2 | Checkpointer good, cleanup/concurrency custom |
| ME: Memory | 8 | 6 | 2 | Store + LangMem strong, cleanup/PII missing |
| MA: Multi-Agent | 5 | 3 | 2 | Supervisor pattern works, no native delegation |
| GV: Governance | 6 | 1 | 5 | interrupt() enables gates, policy/audit custom |
| DR: Determinism & Replay | 6 | 1 | 5 | Checkpointer partial replay, idempotency custom |
| CX: Connectors & Ops | 4 | 0 | 4 | State Migration weak, others custom |
| OB: Observability | 7 | 3 | 4 | LangSmith available but not OTel compliant |
| TE: Testing & Evaluation | 5 | 3 | 2 | Graph structure enables mocking, simulation custom |
| **Total** | **57** | **28** | **29** | |

### Fail-Close Items Status

| Item | Rating | Impact | Applies To |
|------|--------|--------|------------|
| TE-01 Unit Test / Mocking | ⭐⭐⭐⭐ | **PASS** - Graph structure enables node-level mocking | All Authority |
| GV-01 Destructive Operation Gate | ⭐⭐⭐ | **PASS** - interrupt() provides mechanism | Restricted Write+ |
| DR-01 Replay | ⭐⭐ | **BORDERLINE** - Checkpointer partial, LLM not cached | Restricted Write+ |
| DR-04 Idempotency | ⭐ | **FAIL** - No native support | Full Write |
| CX-02 Rate Limit/Retry | ⭐ | **FAIL** - No native rate limiting | Restricted Write+ |
| OB-01 Trace | ⭐⭐⭐ | **PASS** - LangSmith integration available | Full Write |
| OB-06 SLO / Alerts | ⭐ | **FAIL** - No native SLO management | Full Write |

> **Fail-Close Rule**: When any of these items is ⭐⭐ or below, overall rating cap is ⭐⭐ regardless of other categories.
> TE-01 is required for all Authority levels. Other items apply based on write authority.

---

## Good Items (Rating ⭐⭐⭐ and Above)

| Category | ID | Item | Rating | Notes |
|----------|-----|------|--------|-------|
| Tool Calling | TC-01 | Tool Definition | ⭐⭐⭐⭐⭐ | @tool decorator + StructuredTool, excellent docs |
| Tool Calling | TC-03 | Parallel Execution | ⭐⭐⭐⭐⭐ | Multiple tool calls in one AIMessage |
| Tool Calling | TC-04 | Error Handling | ⭐⭐⭐ | handle_tool_errors=True, retry is custom |
| Tool Calling | TC-05 | Argument Validation | ⭐⭐⭐⭐ | Pydantic args_schema native support |
| Human-in-the-Loop | HI-01 | Interrupt API | ⭐⭐⭐⭐⭐ | interrupt() is clean and intuitive |
| Human-in-the-Loop | HI-02 | State Manipulation | ⭐⭐⭐⭐ | Full state access via get_state() |
| Human-in-the-Loop | HI-03 | Resume Control | ⭐⭐⭐⭐⭐ | Command(resume=...) with goto/update |
| Durable Execution | DU-01 | State Persistence | ⭐⭐⭐⭐ | Postgres/SQLite checkpointers |
| Durable Execution | DU-02 | Process Resume | ⭐⭐⭐⭐ | Resume via thread_id works |
| Durable Execution | DU-03 | HITL Persistence | ⭐⭐⭐⭐⭐ | Interrupts survive restart |
| Durable Execution | DU-04 | Storage Options | ⭐⭐⭐⭐ | Memory/SQLite/Postgres |
| Memory | ME-01 | Short-term Memory | ⭐⭐⭐⭐ | add_messages reducer |
| Memory | ME-02 | Long-term Memory | ⭐⭐⭐⭐ | Store with namespace |
| Memory | ME-03 | Semantic Search | ⭐⭐⭐⭐⭐ | Embedding-based search |
| Memory | ME-04 | Memory API | ⭐⭐⭐⭐⭐ | put/get/search/delete |
| Memory | ME-05 | Agent Autonomous Management | ⭐⭐⭐⭐⭐ | LangMem tools (unique feature) |
| Memory | ME-06 | Auto Extraction | ⭐⭐⭐⭐ | LangMem background extraction |
| Multi-Agent | MA-03 | Hierarchical Process | ⭐⭐⭐⭐ | Supervisor pattern implementation |
| Multi-Agent | MA-04 | Routing | ⭐⭐⭐⭐ | Conditional edges, flexible |
| Multi-Agent | MA-05 | Shared Memory | ⭐⭐⭐⭐ | Store shared across threads |
| Governance | GV-01 | Destructive Operation Gate | ⭐⭐⭐ | interrupt() + custom policy engine |
| Observability | OB-01 | Trace | ⭐⭐⭐ | LangSmith integration available |
| Observability | OB-02 | Token Consumption | ⭐⭐⭐⭐ | Via LangSmith or response metadata |
| Observability | OB-04 | External Integration | ⭐⭐⭐⭐ | LangSmith native support |
| Testing & Evaluation | TE-01 | Unit Test / Mocking | ⭐⭐⭐⭐ | Graph structure enables node-level mocking, LLM injectable |
| Testing & Evaluation | TE-02 | Test Fixtures / State Injection | ⭐⭐⭐⭐ | Checkpointer allows state injection for testing |
| Testing & Evaluation | TE-05 | Evaluation Hooks | ⭐⭐⭐ | Node-based hooks possible, custom implementation |

---

## Not Good Items (Rating ⭐⭐ and Below)

| Category | ID | Item | Rating | Notes | Verification Script |
|----------|-----|------|--------|-------|---------------------|
| Tool Calling | TC-02 | Controllable Automation | ⭐⭐ | No native policy control | 19_governance_gate.py |
| Human-in-the-Loop | HI-04 | Timeout | ⭐ | No native timeout | 16_production_considerations.py |
| Human-in-the-Loop | HI-05 | Notification | ⭐ | No native notification | 16_production_considerations.py |
| Durable Execution | DU-05 | Cleanup (TTL) | ⭐ | No auto-cleanup | 09_durable_production.py |
| Durable Execution | DU-06 | Concurrent Access | ⭐⭐ | Race condition on same thread_id | 09_durable_production.py |
| Memory | ME-07 | Memory Cleanup (TTL) | ⭐ | No native memory TTL | 16_production_considerations.py |
| Memory | ME-08 | Embedding Cost | ⭐⭐ | Per-operation cost, no tracking | 16_production_considerations.py |
| Multi-Agent | MA-01 | Multiple Agent Definition | ⭐⭐⭐ | Programmatic, no declarative config | 17_multiagent_supervisor.py |
| Multi-Agent | MA-02 | Delegation | ⭐⭐ | Manual handoff tools required | 18_multiagent_swarm.py |
| Governance | GV-02 | Least Privilege / Scope | ⭐ | No native permission system | 20_governance_policy.py |
| Governance | GV-03 | Policy as Code | ⭐ | No native policy engine | 20_governance_policy.py |
| Governance | GV-04 | PII / Redaction | ⭐ | No native redaction | 21_governance_audit.py |
| Governance | GV-05 | Tenant / Purpose Binding | ⭐ | No native purpose binding | - |
| Governance | GV-06 | Audit Trail Completeness | ⭐ | No native audit logging | 21_governance_audit.py |
| Determinism & Replay | DR-02 | Evidence Reference | ⭐ | No native evidence collection | 23_determinism_evidence.py |
| Determinism & Replay | DR-03 | Non-determinism Isolation | ⭐ | No native LLM isolation mode | 23_determinism_evidence.py |
| Determinism & Replay | DR-04 | Idempotency | ⭐ | No native idempotency keys | 22_determinism_replay.py |
| Determinism & Replay | DR-05 | Plan Diff | ⭐ | No native diff visualization | 24_determinism_recovery.py |
| Determinism & Replay | DR-06 | Failure Recovery | ⭐ | No native recovery mechanism | 24_determinism_recovery.py |
| Connectors & Ops | CX-01 | Auth / Credential Management | ⭐ | No native OAuth/token management | 25_connectors_auth.py |
| Connectors & Ops | CX-02 | Rate Limit / Retry | ⭐ | No native rate limiting | 26_connectors_ratelimit.py |
| Connectors & Ops | CX-03 | Async Job Pattern | ⭐ | No native job tracking | 27_connectors_async.py |
| Connectors & Ops | CX-04 | State Migration | ⭐⭐ | Schema changes break old checkpoints, no migration support | 27_connectors_async.py |
| Observability | OB-03 | Log Output | ⭐⭐ | No native structured logging | - |
| Observability | OB-05 | OTel Compliance | ⭐ | No native OpenTelemetry support | 28_observability_otel.py |
| Observability | OB-06 | SLO / Alerts | ⭐ | No native SLO management | 29_observability_guard.py |
| Observability | OB-07 | Cost Guard | ⭐ | No native budget/kill switch | 29_observability_guard.py |
| Testing & Evaluation | TE-03 | Simulation / User Emulation | ⭐⭐ | No native user simulation, custom implementation | - |
| Testing & Evaluation | TE-04 | Dry Run / Sandbox Mode | ⭐⭐ | No native dry run mode, custom tool wrapper needed | - |

---

## Verification Scripts

| Script | Categories | Key Verification Items |
|--------|------------|------------------------|
| 01_quickstart.py | - | Basic LangGraph structure |
| 02_tool_definition.py | Tool Calling | TC-01: @tool, StructuredTool, args_schema |
| 03_tool_execution.py | Tool Calling | TC-01, TC-03: ToolNode, parallel calls |
| 04_tool_error_handling.py | Tool Calling | TC-04: Error handling, retry |
| 05_hitl_interrupt.py | Human-in-the-Loop | HI-01, HI-02: interrupt(), get_state() |
| 06_hitl_approve_reject_edit.py | Human-in-the-Loop | HI-03: Command(resume=...) patterns |
| 07_durable_basic.py | Durable Execution | DU-01: Checkpoint behavior |
| 08_durable_hitl.py | Durable Execution | DU-03: HITL persistence |
| 09_durable_production.py | Durable Execution | DU-05, DU-06: Cleanup, concurrency |
| 11_memory_store_basic.py | Memory | ME-01, ME-04: Store CRUD |
| 12_memory_semantic_search.py | Memory | ME-03: Semantic search |
| 13_memory_cross_thread.py | Memory | ME-02: Cross-thread persistence |
| 14_memory_langmem_tools.py | Memory | ME-05: Agent memory tools |
| 15_memory_background_extraction.py | Memory | ME-06: Auto extraction |
| 16_production_considerations.py | Human-in-the-Loop, Durable Execution, Memory | Overall production summary |
| 17_multiagent_supervisor.py | Multi-Agent | MA-01, MA-03, MA-05: Supervisor pattern |
| 18_multiagent_swarm.py | Multi-Agent | MA-02, MA-04: Swarm/handoff pattern |
| 19_governance_gate.py | Governance, Tool Calling | GV-01, TC-02: Approval gate |
| 20_governance_policy.py | Governance | GV-02, GV-03: Policy engine |
| 21_governance_audit.py | Governance | GV-04, GV-06: PII, audit trail |
| 22_determinism_replay.py | Determinism & Replay | DR-01, DR-04: Replay, idempotency |
| 23_determinism_evidence.py | Determinism & Replay | DR-02, DR-03: Evidence, determinism |
| 24_determinism_recovery.py | Determinism & Replay | DR-05, DR-06: Plan diff, recovery |
| 25_connectors_auth.py | Connectors & Ops | CX-01: OAuth, secret management |
| 26_connectors_ratelimit.py | Connectors & Ops | CX-02: Rate limiting, circuit breaker |
| 27_connectors_async.py | Connectors & Ops | CX-03, CX-04: Async jobs, schema |
| 28_observability_otel.py | Observability | OB-05: OpenTelemetry integration |
| 29_observability_guard.py | Observability | OB-06, OB-07: SLO, cost guard |

---

# Part 1: Quick Start

## 1.1 Minimal Configuration (01_quickstart.py)

**Goal**: Understand LangGraph basics

### Graph Structure

```
START → chatbot → END
```

### Key Code

```python
# State definition - TypedDict based
class State(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages for appending

# Node definition - function
def chatbot(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # Return diff (merged by add_messages)

# Graph construction
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Execute
result = graph.invoke({"messages": [("user", "Hello")]})
```

### Learnings

| Element | Description |
|---------|-------------|
| `StateGraph(State)` | Initialize graph |
| `add_node(name, fn)` | Add node |
| `add_edge(from, to)` | Connect edges |
| `compile()` | Convert to executable graph |
| Node function | `State → State` (diff return OK) |

---

# Part 2: Tool Calling

## 2.1 Tool Definition Methods (02_tool_definition.py)

**Goal**: Compare tool definition approaches

### Four Definition Methods

```python
# Method 1: @tool decorator (Simple)
@tool
def get_weather_simple(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"

# Method 2: @tool with Annotated (Better docs)
@tool
def get_weather_typed(
    city: Annotated[str, "The city name to get weather for"],
    unit: Annotated[str, "Temperature unit"] = "celsius"
) -> str:
    """Get current weather for a city with specified unit."""
    ...

# Method 3: Pydantic schema (Full control)
class WeatherInput(BaseModel):
    city: str = Field(description="The city name")
    unit: str = Field(default="celsius", description="Temperature unit")

@tool(args_schema=WeatherInput)
def get_weather_pydantic(city: str, unit: str = "celsius") -> str:
    ...

# Method 4: StructuredTool (Programmatic)
search_tool = StructuredTool.from_function(
    func=_search_impl,
    name="web_search",
    description="Search the web",
    args_schema=SearchInput,
)
```

### Comparison

| Method | Pros | Cons | Recommended For |
|--------|------|------|-----------------|
| @tool simple | Minimal code | No arg descriptions | Prototyping |
| @tool + Annotated | Has descriptions | Verbose for many args | Medium scale |
| @tool + Pydantic | Full control, validation | More boilerplate | Production |
| StructuredTool | Programmatic generation | Most verbose | Dynamic generation |

---

## 2.2 Tool Execution (03_tool_execution.py)

**Goal**: Verify ToolNode behavior

### Graph Structure

```
START → agent → [tool_calls?] → tools → agent → ... → END
                     ↓ no
                    END
```

### Verified Behavior

| Item | Behavior |
|------|----------|
| Single tool | Normal call and response |
| Multiple tools | Parallel calls in one AIMessage |
| Tool selection | LLM selects appropriate tool |
| No data | Returns via ToolMessage, LLM responds appropriately |

---

## 2.3 Error Handling (04_tool_error_handling.py)

**Goal**: Verify behavior when tools fail

### Summary

| Item | Default Behavior |
|------|------------------|
| Exception catch | ✅ Automatic |
| ToolMessage conversion | ✅ Returns error content as ToolMessage |
| Exception propagation | ❌ Does not propagate (graph continues) |
| Retry | ❌ None (custom implementation needed) |
| LLM reaction | Recognizes error and responds appropriately |

---

# Part 3: Human-in-the-Loop (HITL)

## 3.1 Interrupt Basics (05_hitl_interrupt.py)

**Goal**: Verify interrupt/resume behavior

### Core APIs

| API | Role |
|-----|------|
| `interrupt(value)` | Pause graph execution, return value |
| `Command(resume=data)` | Resume interrupted graph, pass data as interrupt() return value |
| `graph.get_state(config)` | Get current state (`state.next` shows interrupt position) |
| Checkpointer | State persistence (required for interrupt) |

---

## 3.2 Approve / Reject / Edit (06_hitl_approve_reject_edit.py)

**Goal**: Verify three approval patterns

### Summary

| Pattern | Behavior | LLM Response |
|---------|----------|--------------|
| Approve | Execute tool → report completion | "Sent successfully" |
| Reject | Skip tool → return to agent | Understands rejection, offers alternatives |
| Edit | Rewrite args → execute tool | Reports completion with edited values |

---

# Part 4: Durable Execution

## 4.1 Basic Checkpoint Behavior (07_durable_basic.py)

**Observation**: Checkpoint saved AFTER each node completes.

## 4.2 HITL + Durability (08_durable_hitl.py)

**Key Finding**: HITL interrupts are fully durable. Server can restart without losing pending approvals.

## 4.3 Production Concerns (09_durable_production.py)

| Concern | Status | Solution |
|---------|--------|----------|
| Checkpoint timing | ✅ After each node | - |
| Resume after restart | ✅ Works | Use same thread_id |
| HITL persistence | ✅ Full support | - |
| Concurrent access | ⚠️ Race condition | Unique thread_id |
| Checkpoint cleanup | ❌ No auto-cleanup | Custom job |
| Thread listing | ❌ No API | Query storage |
| State migration | ⚠️ Manual | Version schema |

---

# Part 5: Memory

## 5.1-5.5 Memory Features

| Feature | Support | Notes |
|---------|---------|-------|
| Basic CRUD | ✅ Full | put/get/delete/search |
| Namespace | ✅ Full | Folder-like structure |
| Semantic search | ✅ Full | OpenAI embeddings |
| Cross-thread | ✅ Full | Store shared across threads |
| LangMem tools | ✅ Full | Agent-managed memory (unique) |
| Background extraction | ✅ Full | Auto fact extraction |
| Production storage | ✅ PostgresStore | pgvector for vectors |
| Cleanup | ❌ None | No TTL/auto-cleanup |
| Privacy | ⚠️ Manual | PII handling needed |

---

# Part 6: Production Considerations (16_production_considerations.py)

| Concern | Status | Notes |
|---------|--------|-------|
| Audit logging | ⚠️ Manual | No built-in audit |
| Timeout | ⚠️ Manual | No built-in mechanism |
| Notification | ⚠️ Manual | No built-in system |
| Authorization | ⚠️ Manual | No built-in RBAC |
| Memory cleanup | ⚠️ Manual | No TTL |
| Embedding costs | ⚠️ Manual | Per-operation cost |

---

# Part 7: Multi-Agent (MA)

## 7.1 Supervisor Pattern (17_multiagent_supervisor.py)

**Goal**: Evaluate multi-agent hierarchical coordination

### Architecture

```
START → supervisor → [route] → researcher/writer/reviewer → supervisor → ... → END
```

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| MA-01 Multiple Agent Definition | ⭐⭐⭐ | Programmatic, create_react_agent available |
| MA-03 Hierarchical Process | ⭐⭐⭐⭐ | Supervisor pattern works well |
| MA-05 Shared Memory | ⭐⭐⭐⭐ | Store with namespace isolation |

### Comparison with CrewAI

| Aspect | LangGraph | CrewAI |
|--------|-----------|--------|
| Agent Definition | Programmatic | Declarative (role/goal/backstory) |
| Manager | Custom supervisor node | Process.hierarchical built-in |
| Memory Sharing | Store API | Native agent memory |
| Flexibility | High (more code) | Lower (more abstraction) |

---

## 7.2 Swarm/Handoff Pattern (18_multiagent_swarm.py)

**Goal**: Evaluate agent-to-agent delegation

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| MA-02 Delegation | ⭐⭐ | Manual handoff tools required |
| MA-04 Routing | ⭐⭐⭐⭐ | Conditional edges, flexible routing |

### Key Finding

CrewAI's `allow_delegation=True` is one line; LangGraph requires manual handoff tool creation and routing logic. LangGraph is more flexible but more verbose.

---

# Part 8: Governance (GV)

## 8.1 Approval Gate (19_governance_gate.py)

**Goal**: Evaluate destructive operation gate (GV-01, TC-02)

### Implementation

```python
def approval_gate(state: State) -> Command:
    tool_call = state["messages"][-1].tool_calls[0]

    risk_level, requires_approval, reason = policy_engine.evaluate(
        tool_call["name"], tool_call["args"]
    )

    if requires_approval:
        decision = interrupt({
            "action": "approve_destructive",
            "tool_name": tool_call["name"],
            "risk_level": risk_level.value,
        })

        if decision.get("action") != "approve":
            return Command(goto="agent", update={"messages": [rejection_msg]})

    return Command(goto="tools")
```

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| GV-01 Destructive Operation Gate | ⭐⭐⭐ | interrupt() provides mechanism, policy custom |
| TC-02 Controllable Automation | ⭐⭐ | No native policy control |

---

## 8.2 Policy Engine (20_governance_policy.py)

**Goal**: Evaluate least privilege and policy as code (GV-02, GV-03)

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| GV-02 Least Privilege | ⭐ | No native permission system |
| GV-03 Policy as Code | ⭐ | No native policy engine |

### Custom Implementation Required

- PermissionManager with scopes
- PolicyEngine with rule evaluation
- Principal/identity management

---

## 8.3 Audit Trail & PII (21_governance_audit.py)

**Goal**: Evaluate audit logging and PII redaction (GV-04, GV-06)

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| GV-04 PII / Redaction | ⭐ | No native redaction |
| GV-06 Audit Trail | ⭐ | No native audit logging |

### Custom Implementation Required

- PIIRedactor with regex patterns
- AuditTrail with hash chain for tamper detection

---

# Part 9: Determinism & Replay (DR)

## 9.1 Replay & Idempotency (22_determinism_replay.py)

**Goal**: Evaluate replay and exactly-once execution (DR-01, DR-04)

### Checkpointer-Based Replay

```python
def replay_from_checkpoint(graph, config: dict, target_step: int):
    states = list(graph.get_state_history(config))
    target_state = states[-(target_step + 1)]

    replay_config = {
        **config,
        "configurable": {
            **config["configurable"],
            "checkpoint_id": target_state.config["configurable"]["checkpoint_id"]
        }
    }
    return graph.invoke(None, config=replay_config)
```

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| DR-01 Replay | ⭐⭐ | Checkpointer partial, LLM responses not cached |
| DR-04 Idempotency | ⭐ | No native idempotency key support |

### Key Finding

Checkpointer captures state snapshots but doesn't record LLM responses. Replay restarts execution from a checkpoint but may produce different outputs due to LLM non-determinism.

---

## 9.2 Evidence Collection (23_determinism_evidence.py)

**Goal**: Evaluate evidence reference and non-determinism isolation (DR-02, DR-03)

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| DR-02 Evidence Reference | ⭐ | No native evidence collection |
| DR-03 Non-determinism Isolation | ⭐ | No native LLM isolation mode |

---

## 9.3 Recovery (24_determinism_recovery.py)

**Goal**: Evaluate plan diff and failure recovery (DR-05, DR-06)

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| DR-05 Plan Diff | ⭐ | No native diff visualization |
| DR-06 Failure Recovery | ⭐ | No native rollback/compensate |

---

# Part 10: Connectors (CX)

## 10.1 Authentication (25_connectors_auth.py)

**Goal**: Evaluate OAuth and credential management (CX-01)

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| CX-01 Auth / Credential Management | ⭐ | No native OAuth/secret management |

### Custom Implementation Required

- SecretManager for credential storage
- OAuthClient for token management
- CredentialProvider for just-in-time injection

---

## 10.2 Rate Limiting (26_connectors_ratelimit.py)

**Goal**: Evaluate rate limiting and retry (CX-02)

### Implementation Pattern

```python
class RateLimitedToolNode:
    def __init__(self, tools: list, rate_limiter: RateLimiter):
        self.tool_node = ToolNode(tools)
        self.rate_limiter = rate_limiter

    def __call__(self, state: State) -> State:
        tool_call = state["messages"][-1].tool_calls[0]
        if not self.rate_limiter.acquire(tool_call["name"]):
            return {"messages": [ToolMessage(content="Rate limit exceeded", ...)]}
        return self.tool_node(state)
```

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| CX-02 Rate Limit / Retry | ⭐ | No native rate limiting |

### Custom Implementation Required

- TokenBucket for rate limiting
- ExponentialBackoff for retry delays
- CircuitBreaker for fail-fast pattern

---

## 10.3 Async Jobs & State Migration (27_connectors_async.py)

**Goal**: Evaluate async job pattern and state migration (CX-03, CX-04)

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| CX-03 Async Job Pattern | ⭐ | No native job tracking |
| CX-04 State Migration | ⭐⭐ | Schema changes break old checkpoints, no migration mapper |

### Key Finding (CX-04 State Migration)

When state schema is modified (e.g., adding a new field), old checkpoints cannot be loaded without manual migration. This is a critical gap for long-running agents where code versions change during execution.

---

# Part 11: Observability (OB)

## 11.1 OpenTelemetry (28_observability_otel.py)

**Goal**: Evaluate OTel GenAI Semantic Conventions compliance (OB-05)

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| OB-05 OTel Compliance | ⭐ | No native OTel support |

### Key Finding

LangSmith provides observability but is vendor-specific. OTel GenAI Semantic Conventions require custom instrumentation.

---

## 11.2 SLO & Cost Guard (29_observability_guard.py)

**Goal**: Evaluate SLO management and cost guard (OB-06, OB-07)

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| OB-06 SLO / Alerts | ⭐ | No native SLO management |
| OB-07 Cost Guard | ⭐ | No native budget/kill switch |

### Custom Implementation Required

- MetricsCollector for latency/error tracking
- SLOManager for SLO definition and monitoring
- CostGuard for budget tracking and kill switch

---

# Part 12: Testing & Evaluation (TE)

## 12.1 Unit Test / Mocking

**Goal**: Evaluate testability of LangGraph agents

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| TE-01 Unit Test / Mocking | ⭐⭐⭐⭐ | Graph structure allows node-level testing with mock LLMs |

### Key Finding

LangGraph's graph architecture enables clean separation of concerns:
- Nodes can be tested individually with mock inputs
- LLM can be replaced with a fake LLM (e.g., `FakeListChatModel`)
- State transitions can be verified independently

```python
# Mock LLM injection example
from langchain_core.language_models.fake_chat_models import FakeListChatModel

fake_llm = FakeListChatModel(responses=["Fixed response"])
# Inject into graph node for deterministic testing
```

---

## 12.2 Test Fixtures / State Injection

**Goal**: Evaluate state injection for mid-point testing

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| TE-02 Test Fixtures / State Injection | ⭐⭐⭐⭐ | Checkpointer enables loading specific states for testing |

### Key Finding

Checkpointer's `get_state_history()` allows loading any historical state, enabling "5th turn only" testing without executing turns 1-4.

---

## 12.3 Simulation / User Emulation

**Goal**: Evaluate automated agent testing with virtual users

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| TE-03 Simulation / User Emulation | ⭐⭐ | No native simulation, custom implementation required |

---

## 12.4 Dry Run / Sandbox Mode

**Goal**: Evaluate tool execution without side effects

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| TE-04 Dry Run / Sandbox Mode | ⭐⭐ | No native dry run mode, tool wrapper needed |

---

## 12.5 Evaluation Hooks

**Goal**: Evaluate integration of eval functions into execution pipeline

### Evaluation

| Item | Rating | Notes |
|------|--------|-------|
| TE-05 Evaluation Hooks | ⭐⭐⭐ | Node-based architecture allows custom eval nodes |

### Key Finding

LangGraph's node-based architecture allows inserting evaluation nodes between steps, but no native eval framework integration (LangSmith SDK provides partial support).

---

# CrewAI Comparison

| Aspect | LangGraph | CrewAI |
|--------|-----------|--------|
| Learning Curve | Steep (low-level) | Gentle (high-level) |
| Abstraction Level | Low (graph primitives) | High (Crew/Agent/Task) |
| Multi-Agent (MA) | ⭐⭐⭐ Custom | ⭐⭐⭐⭐⭐ Native |
| Delegation | ⭐⭐ Manual handoff | ⭐⭐⭐⭐⭐ allow_delegation=True |
| HITL (HI) | ⭐⭐⭐⭐ interrupt() | ⭐⭐⭐ human_input=True |
| Durable (DU) | ⭐⭐⭐⭐ Checkpointer | ⭐⭐⭐⭐ @persist |
| Memory (ME) | ⭐⭐⭐⭐⭐ LangMem (unique) | ⭐⭐⭐ Basic |
| Governance (GV) | ⭐⭐ Custom | ⭐ Custom |
| Determinism (DR) | ⭐⭐ Partial | ⭐ Custom |
| Observability (OB) | ⭐⭐⭐ LangSmith | ⭐⭐ Limited |
| Flexibility | ⭐⭐⭐⭐⭐ Full control | ⭐⭐⭐ Constrained |

---

# Recommendations

## When to Use LangGraph

- Complex workflows requiring fine-grained control
- Custom graph topologies (not just sequential/hierarchical)
- Advanced HITL with approve/reject/edit patterns
- Long-term memory with semantic search (LangMem)
- Integration with LangSmith ecosystem
- Research and experimentation

## When NOT to Use LangGraph (Without Custom Implementation)

- Production systems requiring audit trails
- Regulatory compliance (PII, GDPR)
- High-volume API integrations (rate limiting needed)
- Systems requiring replay/debugging capability
- Teams preferring declarative agent definitions

## Production Deployment Requirements

For production deployment with external writes, implement:

1. **Governance Layer** (Scripts 19-21)
   - ApprovalGate with interrupt()
   - PolicyEngine for least privilege
   - PIIRedactor for sensitive data
   - AuditTrail with hash chain

2. **Determinism Infrastructure** (Scripts 22-24)
   - ReplayLogger for LLM response caching
   - IdempotencyKeyManager for exactly-once
   - RecoveryManager for failure handling

3. **Connector Abstractions** (Scripts 25-27)
   - TokenBucket + ExponentialBackoff
   - OAuthClient + SecretManager
   - AsyncJobExecutor

4. **Observability Stack** (Scripts 28-29)
   - OTel instrumentation
   - SLOManager + CostGuard

---

# Conclusion

## Overall Rating: ⭐⭐⭐ (PoC Ready - with Fail-Close considered)

LangGraph provides excellent low-level primitives but lacks enterprise-grade safety features:

### Strengths (⭐⭐⭐⭐+)

- **HITL**: interrupt()/Command API is elegant and powerful
- **Memory**: Store + LangMem provides unique agent-managed memory
- **Durable Execution**: Checkpointer enables reliable state persistence
- **Tool Calling**: Excellent tool definition and execution support
- **Flexibility**: Full control over graph structure and execution

### Weaknesses (⭐ to ⭐⭐)

- **Governance**: No policy engine, audit trail, or PII handling
- **Determinism**: Partial replay support, no idempotency
- **Connectors**: No rate limiting, OAuth, or async job patterns
- **Observability**: LangSmith-centric, not OTel compliant

### Fail-Close Impact

Due to ⭐ ratings on:
- DR-04 (Idempotency) - Full Write
- CX-02 (Rate Limit/Retry) - Restricted Write+
- OB-06 (SLO/Alerts) - Full Write

**The overall rating is bounded by these critical gaps for production systems with external writes.**

TE-01 (Unit Test/Mocking) passes at ⭐⭐⭐⭐, which is a strength for development workflow.

### vs CrewAI

- **LangGraph** excels at flexibility, HITL, and memory
- **CrewAI** excels at multi-agent collaboration and ease of use
- Both require similar custom work for governance and determinism

### Recommendation

Use LangGraph when you need:
1. Complex graph topologies
2. Advanced HITL patterns
3. LangMem for agent-managed memory
4. Full control over execution flow

Build the surrounding infrastructure using the verification scripts (17-29) as templates.

---

# File Structure

```
lang-graph-sample/
├── 01_quickstart.py              # Quick Start
├── 02_tool_definition.py         # Tool definition comparison
├── 03_tool_execution.py          # ToolNode verification
├── 04_tool_error_handling.py     # Error handling
├── 05_hitl_interrupt.py          # HITL basics (interrupt)
├── 06_hitl_approve_reject_edit.py # Approve/Reject/Edit
├── 07_durable_basic.py           # Durable execution basics
├── 08_durable_hitl.py            # HITL + Durability
├── 09_durable_production.py      # Durable production concerns
├── 11_memory_store_basic.py      # Memory Store CRUD
├── 12_memory_semantic_search.py  # Semantic search
├── 13_memory_cross_thread.py     # Cross-thread persistence
├── 14_memory_langmem_tools.py    # LangMem agent tools
├── 15_memory_background_extraction.py # Background extraction
├── 16_production_considerations.py # Overall production summary
├── 17_multiagent_supervisor.py   # Supervisor pattern
├── 18_multiagent_swarm.py        # Swarm/handoff pattern
├── 19_governance_gate.py         # Approval gate
├── 20_governance_policy.py       # Policy engine
├── 21_governance_audit.py        # Audit trail & PII
├── 22_determinism_replay.py      # Replay & idempotency
├── 23_determinism_evidence.py    # Evidence collection
├── 24_determinism_recovery.py    # Plan diff & recovery
├── 25_connectors_auth.py         # OAuth & credentials
├── 26_connectors_ratelimit.py    # Rate limiting
├── 27_connectors_async.py        # Async jobs & schema
├── 28_observability_otel.py      # OpenTelemetry
├── 29_observability_guard.py     # SLO & cost guard
├── REPORT.md                     # This report
├── REPORT_ja.md                  # Japanese version
├── .env.example                  # Environment template
├── pyproject.toml
└── uv.lock
```
