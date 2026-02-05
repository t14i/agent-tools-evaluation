# Inngest Durable Execution Verification Report

## Overview

This report summarizes the findings from evaluating Inngest (Python SDK) for production readiness based on the Durable Execution Evaluation Criteria (ExoFlow OSDI'23, Flux OSDI'23, Temporal 4 Properties, Restate, NIST AI RMF, OTel Semantic Conventions).

## Test Environment

- Python: 3.13
- Inngest SDK: 0.5.x
- Inngest Dev Server: latest
- OS: macOS (Darwin)

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
| Duration | Minutes to Hours |
| Side Effects | Restricted Write |
| Deployment | Serverless / Cloud |
| Scale | ~100 |
| Determinism | LLM Decision Involved |
| Language | Python |

---

## Evaluation Summary by Category

### Coverage Summary (58 Items)

| Category | Items | Good (⭐⭐⭐+) | Not Good (⭐⭐-) | Notes |
|----------|-------|---------------|-----------------|-------|
| EX: Execution Semantics | 7 | 6 | 1 | Journal-based memoization excellent, determinism detection weaker |
| RT: Retry & Timeout | 4 | 2 | 2 | Retry good, timeout/heartbeat limited |
| WF: Workflow Primitives | 6 | 6 | 0 | Full workflow constructs |
| SG: Signals & Events | 4 | 3 | 1 | Event-driven native, query limited |
| VR: Versioning & Migration | 4 | 2 | 2 | Basic versioning, detection weak |
| CP: Compensation & Recovery | 4 | 4 | 0 | Manual saga, good recovery |
| PF: Performance & Overhead | 3 | 2 | 1 | HTTP overhead, good throughput |
| OP: Operations | 7 | 7 | 0 | Excellent serverless deployment |
| OB: Observability | 6 | 5 | 1 | Good UI, OTel limited |
| DX: Developer Experience | 7 | 5 | 2 | Simple API, no time skipping |
| AI: AI/Agent Integration | 6 | 5 | 1 | AgentKit excellent, streaming limited |
| **Total** | **58** | **47** | **11** | |

### Fail-Close Items Status

| Item | Rating | Impact | Applies To |
|------|--------|--------|------------|
| EX-01 Progress Guarantee | ⭐⭐⭐⭐⭐ | **PASS** - Journal-based step memoization | Always Required |
| EX-03 Idempotency / Deduplication | ⭐⭐⭐⭐⭐ | **PASS** - Event-level idempotency key | Always Required |
| RT-01 Retry Strategy | ⭐⭐⭐⭐⭐ | **PASS** - Function retries + RetryAfterError | Always Required |
| VR-01 Workflow Versioning | ⭐⭐⭐ | **PASS** - function_id versioning (basic) | Duration >= Minutes |
| OP-02 Workflow Management API | ⭐⭐⭐⭐ | **PASS** - List/Get/Replay/Cancel | Always Required |

> **Fail-Close Rule**: All fail-close items pass at ⭐⭐⭐ or above. No overall rating cap applied.

---

## Good Items (Rating ⭐⭐⭐ and Above)

| Category | ID | Item | Rating | Notes |
|----------|-----|------|--------|-------|
| Execution Semantics | EX-01 | Progress Guarantee | ⭐⭐⭐⭐⭐ | Journal-based step memoization ensures progress |
| Execution Semantics | EX-02 | Side Effect Guarantee | ⭐⭐⭐⭐ | step.run() memoizes results, external idempotency user responsibility |
| Execution Semantics | EX-03 | Idempotency / Deduplication | ⭐⭐⭐⭐⭐ | Event-level idempotency key native support |
| Execution Semantics | EX-04 | State Persistence | ⭐⭐⭐⭐⭐ | Journal-based persistence, simpler than Event Sourcing |
| Execution Semantics | EX-05 | Determinism Constraints | ⭐⭐⭐⭐ | Relaxed constraints - only step boundaries matter, but step ID changes can cause issues |
| Execution Semantics | EX-07 | Replay Accuracy | ⭐⭐⭐⭐ | Step memoization accurate, less granular than Temporal |
| Retry & Timeout | RT-01 | Retry Strategy | ⭐⭐⭐⭐⭐ | Exponential backoff, jitter, RetryAfterError, NonRetriableError |
| Retry & Timeout | RT-03 | Circuit Breaker | ⭐⭐⭐ | max_attempts + NonRetriableError achieves similar result |
| Workflow Primitives | WF-01 | Step Definition | ⭐⭐⭐⭐⭐ | step.run() simple and effective |
| Workflow Primitives | WF-02 | Child Workflows | ⭐⭐⭐⭐ | step.invoke() for child function calls |
| Workflow Primitives | WF-03 | Parallel Execution / Fan-out | ⭐⭐⭐⭐⭐ | step.parallel() for natural parallel execution |
| Workflow Primitives | WF-04 | Conditional / Loop | ⭐⭐⭐⭐⭐ | Standard Python control flow |
| Workflow Primitives | WF-05 | Sleep / Timer | ⭐⭐⭐⭐⭐ | Durable sleep, Cron triggers |
| Workflow Primitives | WF-06 | Queue / Rate Control | ⭐⭐⭐⭐ | Concurrency limits, rate limits |
| Signals & Events | SG-01 | External Signals | ⭐⭐⭐⭐⭐ | step.send_event() for event-driven communication |
| Signals & Events | SG-02 | Wait / Awaitables | ⭐⭐⭐⭐⭐ | step.wait_for_event() with expression matching |
| Signals & Events | SG-03 | Event Triggers | ⭐⭐⭐⭐⭐ | Event-driven is native architecture |
| Versioning & Migration | VR-01 | Workflow Versioning | ⭐⭐⭐ | function_id versioning, manual drain waiting for in-flight runs |
| Versioning & Migration | VR-04 | Schema Evolution | ⭐⭐⭐⭐ | JSON flexibility, Pydantic validation |
| Compensation & Recovery | CP-01 | Compensation / Saga | ⭐⭐⭐⭐ | Manual try/except + compensation pattern |
| Compensation & Recovery | CP-02 | Partial Resume | ⭐⭐⭐⭐⭐ | Automatic partial resume via step memoization |
| Compensation & Recovery | CP-03 | Manual Intervention | ⭐⭐⭐⭐ | Dashboard/API Replay/Cancel |
| Compensation & Recovery | CP-04 | Dead Letter | ⭐⭐⭐⭐ | Failed status visible, NonRetriableError |
| Performance & Overhead | PF-02 | Fan-out Throughput | ⭐⭐⭐⭐ | step.parallel() with good throughput |
| Performance & Overhead | PF-03 | Payload Size Limits | ⭐⭐⭐⭐ | 512KB limit, external storage pattern |
| Operations | OP-01 | Deployment Model | ⭐⭐⭐⭐⭐ | Serverless-native, Cloud/Self-hosted options |
| Operations | OP-02 | Workflow Management API | ⭐⭐⭐⭐ | List/Get/Replay/Cancel |
| Operations | OP-03 | Storage Backend | ⭐⭐⭐⭐ | PostgreSQL + Redis (simpler than Temporal) |
| Operations | OP-04 | Scalability | ⭐⭐⭐⭐⭐ | Serverless auto-scaling, concurrency limits |
| Operations | OP-05 | Data Retention / Cleanup | ⭐⭐⭐⭐ | Plan-based retention, auto cleanup |
| Operations | OP-06 | Multi-Region / HA | ⭐⭐⭐⭐ | Cloud HA, self-hosted requires external setup |
| Operations | OP-07 | Multi-Tenant Isolation | ⭐⭐⭐⭐ | App ID + concurrency key isolation |
| Observability | OB-01 | Dashboard / UI | ⭐⭐⭐⭐⭐ | Excellent dev server and cloud UI |
| Observability | OB-02 | Metrics | ⭐⭐⭐⭐ | Dashboard metrics, Prometheus for self-hosted |
| Observability | OB-03 | History Visualization | ⭐⭐⭐⭐⭐ | Step timeline, input/output inspection |
| Observability | OB-05 | Alerts | ⭐⭐⭐⭐ | Cloud alerts, webhook integration |
| Observability | OB-06 | Logging | ⭐⭐⭐⭐ | Standard Python logging with context |
| Developer Experience | DX-01 | SDK Design | ⭐⭐⭐⭐⭐ | Simple decorator API, async/await native |
| Developer Experience | DX-02 | Language Support | ⭐⭐⭐⭐ | TypeScript/Python/Go with good parity |
| Developer Experience | DX-03 | Local Development | ⭐⭐⭐⭐⭐ | npx inngest-cli dev, hot reload |
| Developer Experience | DX-05 | Error Messages / Debugging | ⭐⭐⭐⭐ | Clear errors, stack traces, UI debugging |
| Developer Experience | DX-06 | Learning Curve | ⭐⭐⭐⭐⭐ | Simple mental model, relaxed determinism |
| AI/Agent Integration | AI-01 | LLM Call as Activity | ⭐⭐⭐⭐⭐ | step.run() for LLM, results memoized |
| AI/Agent Integration | AI-02 | Non-determinism Handling | ⭐⭐⭐⭐ | Step isolation handles non-determinism |
| AI/Agent Integration | AI-03 | HITL / Human Approval | ⭐⭐⭐⭐⭐ | step.wait_for_event() with timeout |
| AI/Agent Integration | AI-05 | Agent Framework Integration | ⭐⭐⭐⭐⭐ | AgentKit native (experimental, API may change), LangGraph/CrewAI patterns |
| AI/Agent Integration | AI-06 | Tool Execution Fault Tolerance | ⭐⭐⭐⭐⭐ | Per-tool step isolation with retry |

---

## Not Good Items (Rating ⭐⭐ and Below)

| Category | ID | Item | Rating | Notes | Verification Script |
|----------|-----|------|--------|-------|---------------------|
| Execution Semantics | EX-06 | Determinism Violation Handling | ⭐⭐⭐ | No automatic detection, step ID mismatch causes runtime error | 01_ex_execution_semantics.py |
| Retry & Timeout | RT-02 | Timeout System | ⭐⭐⭐ | Simpler model, no fine-grained timeouts like Temporal | 02_rt_retry_timeout.py |
| Retry & Timeout | RT-04 | Heartbeat | ⭐⭐ | No heartbeat mechanism, use step splitting | 02_rt_retry_timeout.py |
| Signals & Events | SG-04 | Query | ⭐⭐⭐ | No native query, use Run API or external state | 04_sg_signals_events.py |
| Versioning & Migration | VR-02 | Breaking Change Detection | ⭐⭐ | No automatic detection or replay tests | 05_vr_versioning.py |
| Versioning & Migration | VR-03 | Migration Strategy | ⭐⭐⭐ | Manual blue-green, no auto drain | 05_vr_versioning.py |
| Performance & Overhead | PF-01 | Step Latency | ⭐⭐⭐ | HTTP overhead ~10-50ms per step | 07_pf_performance.py |
| Observability | OB-04 | OTel Compliance | ⭐⭐⭐ | OTel available but requires manual setup | 09_ob_observability.py |
| Developer Experience | DX-04 | Testing / Time Skipping | ⭐⭐ | No time skipping, long sleeps require real time | 10_dx_developer_experience.py |
| Developer Experience | DX-07 | Local Replay Harness | ⭐⭐ | Cannot replay production history locally | 10_dx_developer_experience.py |
| AI/Agent Integration | AI-04 | Streaming | ⭐⭐⭐ | Cannot stream through Inngest boundary | 11_ai_agent_integration.py |

---

## Verification Scripts

| Script | Categories | Key Verification Items |
|--------|------------|------------------------|
| 01_ex_execution_semantics.py | Execution Semantics | EX-01 to EX-07: Progress, idempotency, replay |
| 02_rt_retry_timeout.py | Retry & Timeout | RT-01 to RT-04: Retry, timeout, heartbeat |
| 03_wf_workflow_primitives.py | Workflow Primitives | WF-01 to WF-06: Steps, child WF, parallel, sleep |
| 04_sg_signals_events.py | Signals & Events | SG-01 to SG-04: Signals, wait, query |
| 05_vr_versioning.py | Versioning & Migration | VR-01 to VR-04: Versioning, migration |
| 06_cp_compensation.py | Compensation & Recovery | CP-01 to CP-04: Saga, resume, intervention |
| 07_pf_performance.py | Performance & Overhead | PF-01 to PF-03: Latency, throughput, payload |
| 08_op_operations.py | Operations | OP-01 to OP-07: Deploy, management, scale |
| 09_ob_observability.py | Observability | OB-01 to OB-06: UI, metrics, OTel, logging |
| 10_dx_developer_experience.py | Developer Experience | DX-01 to DX-07: SDK, testing, debugging |
| 11_ai_agent_integration.py | AI/Agent Integration | AI-01 to AI-06: LLM, HITL, tools |

---

## Key Findings

### Strengths (⭐⭐⭐⭐+)

1. **Developer Experience** (DX: Average ⭐⭐⭐⭐)
   - Simpler mental model than Temporal
   - Relaxed determinism constraints
   - "Normal code" + step boundaries
   - Excellent local development with dev server

2. **Operations** (OP: Average ⭐⭐⭐⭐⭐)
   - Serverless-native architecture
   - Simple deployment model
   - PostgreSQL + Redis (simpler than Temporal)
   - Built-in cloud option

3. **AI/Agent Integration** (AI: Average ⭐⭐⭐⭐⭐)
   - AgentKit for native agent support (provides tool management, state persistence between LLM calls, and automatic retry for tool failures)
   - step.run() for LLM memoization
   - Excellent HITL with wait_for_event
   - LangGraph/CrewAI integration patterns

4. **Workflow Primitives** (WF: Average ⭐⭐⭐⭐⭐)
   - step.run/parallel/sleep/invoke comprehensive
   - Natural Python control flow
   - Durable timers and Cron

5. **Observability** (OB: Average ⭐⭐⭐⭐)
   - Excellent dev server UI
   - Step timeline visualization
   - Cloud dashboard for monitoring

### Considerations

1. **Time Skipping** (DX-04: ⭐⭐)
   - Major gap for testing long-running workflows
   - Workarounds available:
     - Environment variable override for sleep duration
     - Mock/stub pattern for tests
     - Short duration parameter for integration tests
   - Temporal significantly better here

2. **Heartbeat** (RT-04: ⭐⭐)
   - No native heartbeat for long-running steps
   - Must split into multiple steps
   - Temporal's heartbeat + timeout is superior

3. **Versioning** (VR: Average ⭐⭐⭐)
   - Basic function_id versioning
   - No automatic breaking change detection
   - No Build ID routing like Temporal

4. **Local Replay** (DX-07: ⭐⭐)
   - Cannot replay production history locally
   - Manual event copying required
   - Temporal's WorkflowReplayer is far better

5. **Step Latency** (PF-01: ⭐⭐⭐)
   - HTTP-based model adds latency
   - ~10-50ms per step vs Temporal's ~4ms
   - Acceptable for most use cases

---

## Temporal vs Inngest Comparison

| Aspect | Temporal | Inngest |
|--------|----------|---------|
| Execution Model | Event Sourcing + Replay | Journal-based + Step memoization |
| Determinism | Strict (entire workflow) | Relaxed (only step boundaries) |
| Learning Curve | Steeper (concepts) | Gentler (simpler model) |
| Deployment | Worker + Server | Serverless HTTP |
| Time Skipping | ⭐⭐⭐⭐⭐ Built-in | ⭐⭐ Not available |
| Heartbeat | ⭐⭐⭐⭐⭐ Native | ⭐⭐ None |
| Versioning | ⭐⭐⭐⭐⭐ Build ID | ⭐⭐⭐ function_id |
| Local Replay | ⭐⭐⭐⭐⭐ WorkflowReplayer | ⭐⭐ Manual copy |
| Agent Integration | ⭐⭐⭐⭐ Activity pattern | ⭐⭐⭐⭐⭐ AgentKit |
| Infrastructure | Heavier (4 services) | Lighter (Postgres + Redis) |
| Local Dev | ⭐⭐⭐⭐⭐ start-dev | ⭐⭐⭐⭐⭐ inngest-cli dev |
| Cost | Server + storage costs | Pay-per-execution (Cloud) or self-hosted |

### When to Use Inngest

- Simpler deployment requirements
- Serverless/FaaS architecture
- Relaxed determinism constraints OK
- AI/Agent workloads with AgentKit
- Rapid prototyping
- Shorter duration workflows

### When to Use Temporal Instead

- Strict versioning requirements
- Need time skipping for tests
- Long-running activities with heartbeat
- Production history debugging (replay)
- Event Sourcing audit requirements
- Multi-language workflows

---

## Recommendations

### Production Deployment Checklist

For production deployment with Inngest:

1. **Infrastructure**
   - [ ] Inngest Cloud account or self-hosted setup
   - [ ] PostgreSQL for persistence (self-hosted)
   - [ ] Redis for caching (self-hosted)
   - [ ] Function deployment (Vercel/AWS/containers)

2. **Monitoring**
   - [ ] Dashboard access configured
   - [ ] Alert webhooks set up
   - [ ] Prometheus scraping (self-hosted)

3. **Development Practices**
   - [ ] Step IDs are stable and meaningful
   - [ ] Integration tests with dev server
   - [ ] Blue-green deployment for versioning

---

## Conclusion

### Overall Rating: ⭐⭐⭐⭐ (Production Ready)

Inngest is a production-ready durable execution platform that excels in simplicity and AI/Agent integration:

**Strengths:**
- Simplest learning curve among durable execution platforms
- Serverless-native architecture
- Excellent AgentKit for AI workloads
- Good developer experience with dev server
- Relaxed determinism makes "normal code" work

**Weaknesses:**
- No time skipping (significant for long-running workflow tests)
- No heartbeat mechanism
- Basic versioning without automatic detection
- Cannot replay production history locally

### Verdict

Inngest is **recommended** for:
- Teams new to durable execution
- Serverless/FaaS architectures
- AI/Agent applications with HITL
- Shorter duration workflows (minutes to hours)

Consider Temporal instead if you need:
- Time skipping for tests
- Strict versioning and replay debugging
- Long-running activities with heartbeat
- Multi-language support

The 58-item evaluation shows **47 items at ⭐⭐⭐ or above**, with **all 5 fail-close items passing**. Inngest provides an excellent balance of simplicity and durability for modern serverless applications.

---

# File Structure

```
inngest-example/
├── 01_ex_execution_semantics.py   # EX category verification
├── 02_rt_retry_timeout.py         # RT category verification
├── 03_wf_workflow_primitives.py   # WF category verification
├── 04_sg_signals_events.py        # SG category verification
├── 05_vr_versioning.py            # VR category verification
├── 06_cp_compensation.py          # CP category verification
├── 07_pf_performance.py           # PF category verification
├── 08_op_operations.py            # OP category verification
├── 09_ob_observability.py         # OB category verification
├── 10_dx_developer_experience.py  # DX category verification
├── 11_ai_agent_integration.py     # AI category verification
├── serve.py                       # FastAPI server for Inngest functions
├── common.py                      # Shared utilities
├── REPORT.md                      # This report (English)
├── REPORT_ja.md                   # This report (Japanese)
├── README.md                      # Quick start guide
└── pyproject.toml
```
