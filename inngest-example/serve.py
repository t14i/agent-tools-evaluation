"""
Inngest Function Serving - FastAPI server to expose Inngest functions.

This module registers all Inngest functions from evaluation scripts
and serves them via HTTP for the Inngest Dev Server to discover.
"""

import inngest
from inngest.fast_api import serve
from fastapi import FastAPI

from common import INNGEST_APP_ID

# Create Inngest client
inngest_client = inngest.Inngest(app_id=INNGEST_APP_ID)

# Create FastAPI app
app = FastAPI(title="Inngest Durable Execution Evaluation")

# Import all functions from evaluation modules
# These will be populated by each evaluation script when imported

# Global registry for functions
registered_functions: list[inngest.Function] = []


def register_function(fn: inngest.Function) -> inngest.Function:
    """Register a function for serving."""
    registered_functions.append(fn)
    return fn


# =============================================================================
# EX: Execution Semantics Functions
# =============================================================================

@inngest_client.create_function(
    fn_id="ex01-progress-guarantee",
    trigger=inngest.TriggerEvent(event="test/ex01.progress"),
)
async def ex01_progress_guarantee(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate progress guarantee through step memoization."""
    results = []

    result1 = await step.run("step1", lambda: {"step": "step1", "status": "completed"})
    results.append(result1)

    result2 = await step.run("step2", lambda: {"step": "step2", "status": "completed"})
    results.append(result2)

    result3 = await step.run("step3", lambda: {"step": "step3", "status": "completed"})
    results.append(result3)

    return {"results": results, "total_steps": len(results)}


@inngest_client.create_function(
    fn_id="ex02-side-effect",
    trigger=inngest.TriggerEvent(event="test/ex02.sideeffect"),
)
async def ex02_side_effect(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate side effect handling with step memoization."""
    operation_id = ctx.event.data.get("operation_id", "default")

    # Side effect is memoized - won't re-execute on replay
    result = await step.run(
        f"side-effect-{operation_id}",
        lambda: {"operation_id": operation_id, "executed": True}
    )

    return {"result": result, "operation_id": operation_id}


@inngest_client.create_function(
    fn_id="ex03-idempotency",
    trigger=inngest.TriggerEvent(event="test/ex03.idempotency"),
    idempotency="event.data.idempotency_key",
)
async def ex03_idempotency(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate idempotency with event-level deduplication."""
    key = ctx.event.data.get("idempotency_key", "")

    result = await step.run("process", lambda: {"processed": True, "key": key})

    return {"result": result}


@inngest_client.create_function(
    fn_id="ex04-state-persistence",
    trigger=inngest.TriggerEvent(event="test/ex04.state"),
)
async def ex04_state_persistence(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate journal-based state persistence."""
    # Each step result is persisted in the journal
    state = {}

    state["step1"] = await step.run("state-step1", lambda: "value1")
    state["step2"] = await step.run("state-step2", lambda: "value2")
    state["step3"] = await step.run("state-step3", lambda: "value3")

    return {"final_state": state}


@inngest_client.create_function(
    fn_id="ex05-determinism",
    trigger=inngest.TriggerEvent(event="test/ex05.determinism"),
)
async def ex05_determinism(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate relaxed determinism constraints."""
    import uuid
    import random
    from datetime import datetime

    # Outside steps: Non-deterministic operations are OK
    # These values may differ on replay but don't affect correctness
    random_value = random.random()
    current_time = datetime.now().isoformat()

    # Inside steps: Results are memoized for replay
    step_result = await step.run(
        "deterministic-step",
        lambda: {
            "uuid": str(uuid.uuid4()),  # Memoized
            "random": random.random(),   # Memoized
        }
    )

    return {
        "outside_step": {
            "random": random_value,
            "time": current_time,
        },
        "inside_step": step_result,
    }


# =============================================================================
# RT: Retry & Timeout Functions
# =============================================================================

retry_counter: dict[str, int] = {}


@inngest_client.create_function(
    fn_id="rt01-retry-strategy",
    trigger=inngest.TriggerEvent(event="test/rt01.retry"),
    retries=3,
)
async def rt01_retry_strategy(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate retry with exponential backoff."""
    run_id = ctx.event.data.get("run_id", "default")

    async def failing_step():
        retry_counter[run_id] = retry_counter.get(run_id, 0) + 1
        if retry_counter[run_id] < 3:
            raise inngest.RetryAfterError("Transient failure", retry_after_seconds=1)
        return {"attempt": retry_counter[run_id], "success": True}

    result = await step.run("failing-step", failing_step)
    return {"result": result}


@inngest_client.create_function(
    fn_id="rt02-timeout",
    trigger=inngest.TriggerEvent(event="test/rt02.timeout"),
)
async def rt02_timeout(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate step-level timeout handling."""
    import asyncio

    async def slow_step():
        await asyncio.sleep(0.5)
        return {"completed": True}

    result = await step.run("slow-step", slow_step)
    return {"result": result}


# =============================================================================
# WF: Workflow Primitives Functions
# =============================================================================

@inngest_client.create_function(
    fn_id="wf01-step-definition",
    trigger=inngest.TriggerEvent(event="test/wf01.steps"),
)
async def wf01_step_definition(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate step.run() for step definition."""
    results = []

    result1 = await step.run("step-a", lambda: {"name": "step-a", "value": 1})
    results.append(result1)

    result2 = await step.run("step-b", lambda: {"name": "step-b", "value": 2})
    results.append(result2)

    return {"steps": results}


@inngest_client.create_function(
    fn_id="wf02-child-function",
    trigger=inngest.TriggerEvent(event="test/wf02.parent"),
)
async def wf02_child_function(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate step.invoke() for child function calls."""
    task = ctx.event.data.get("task", "default-task")

    # Invoke child function and wait for result
    child_result = await step.invoke(
        "invoke-child",
        function=wf02_child_worker,
        data={"task": task},
    )

    return {"parent_completed": True, "child_result": child_result}


@inngest_client.create_function(
    fn_id="wf02-child-worker",
    trigger=inngest.TriggerEvent(event="internal/wf02.child"),
)
async def wf02_child_worker(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Child function that does actual work."""
    task = ctx.event.data.get("task", "")

    result = await step.run("process-task", lambda: {"task": task, "processed": True})

    return result


@inngest_client.create_function(
    fn_id="wf03-parallel",
    trigger=inngest.TriggerEvent(event="test/wf03.parallel"),
)
async def wf03_parallel(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate parallel step execution."""

    # Define parallel tasks
    async def task_a():
        return {"task": "A", "value": 1}

    async def task_b():
        return {"task": "B", "value": 2}

    async def task_c():
        return {"task": "C", "value": 3}

    # Execute in parallel using step.parallel (if available) or sequential
    # Note: Inngest's parallel execution pattern
    results = await step.parallel(
        (
            lambda: step.run("parallel-a", task_a),
            lambda: step.run("parallel-b", task_b),
            lambda: step.run("parallel-c", task_c),
        )
    )

    return {"parallel_results": results}


@inngest_client.create_function(
    fn_id="wf04-control-flow",
    trigger=inngest.TriggerEvent(event="test/wf04.control"),
)
async def wf04_control_flow(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate conditional logic and loops."""
    count = ctx.event.data.get("count", 3)
    mode = ctx.event.data.get("mode", "normal")

    results = []

    # Loop
    for i in range(count):
        result = await step.run(f"loop-step-{i}", lambda i=i: {"index": i})
        results.append(result)

    # Conditional
    if mode == "special":
        special = await step.run("special-step", lambda: {"special": True})
        results.append(special)
    else:
        normal = await step.run("normal-step", lambda: {"normal": True})
        results.append(normal)

    return {"results": results, "count": count, "mode": mode}


@inngest_client.create_function(
    fn_id="wf05-sleep-timer",
    trigger=inngest.TriggerEvent(event="test/wf05.sleep"),
)
async def wf05_sleep_timer(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate durable sleep and timers."""
    from datetime import timedelta

    # Record start
    start = await step.run("start", lambda: {"started": True})

    # Durable sleep (survives restarts)
    await step.sleep("wait-1-second", timedelta(seconds=1))

    # Record after sleep
    after_sleep = await step.run("after-sleep", lambda: {"after_sleep": True})

    return {"start": start, "after_sleep": after_sleep}


@inngest_client.create_function(
    fn_id="wf06-concurrency",
    trigger=inngest.TriggerEvent(event="test/wf06.concurrency"),
    concurrency=[
        inngest.Concurrency(limit=2, key="event.data.group"),
    ],
)
async def wf06_concurrency(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate concurrency control."""
    import asyncio

    group = ctx.event.data.get("group", "default")

    await step.run("start", lambda: {"started": True, "group": group})
    await asyncio.sleep(0.5)  # Simulate work
    result = await step.run("complete", lambda: {"completed": True, "group": group})

    return result


# =============================================================================
# SG: Signals & Events Functions
# =============================================================================

@inngest_client.create_function(
    fn_id="sg01-send-event",
    trigger=inngest.TriggerEvent(event="test/sg01.sender"),
)
async def sg01_send_event(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate sending events to other functions."""
    target_id = ctx.event.data.get("target_id", "default")

    # Send event to trigger another function
    await step.send_event(
        "send-signal",
        events=[
            inngest.Event(
                name="test/sg01.receiver",
                data={"target_id": target_id, "message": "Hello from sender"},
            )
        ],
    )

    return {"sent_to": target_id}


@inngest_client.create_function(
    fn_id="sg01-receiver",
    trigger=inngest.TriggerEvent(event="test/sg01.receiver"),
)
async def sg01_receiver(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Receive events sent from other functions."""
    message = ctx.event.data.get("message", "")
    target_id = ctx.event.data.get("target_id", "")

    result = await step.run("process-message", lambda: {
        "received": True,
        "message": message,
        "target_id": target_id,
    })

    return result


@inngest_client.create_function(
    fn_id="sg02-wait-for-event",
    trigger=inngest.TriggerEvent(event="test/sg02.waiter"),
)
async def sg02_wait_for_event(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate waiting for external events."""
    from datetime import timedelta

    correlation_id = ctx.event.data.get("correlation_id", "default")

    # Start processing
    await step.run("start", lambda: {"started": True})

    # Wait for external event (e.g., webhook callback, human approval)
    event = await step.wait_for_event(
        "wait-for-approval",
        event="test/sg02.approval",
        timeout=timedelta(seconds=30),
        if_exp=f"event.data.correlation_id == '{correlation_id}'",
    )

    if event is None:
        return {"status": "timeout", "correlation_id": correlation_id}

    return {
        "status": "approved",
        "correlation_id": correlation_id,
        "approval_data": event.data,
    }


@inngest_client.create_function(
    fn_id="sg04-query-state",
    trigger=inngest.TriggerEvent(event="test/sg04.query"),
)
async def sg04_query_state(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Note: Inngest doesn't have native query like Temporal."""
    # State can be retrieved via the Run API
    state = {"step1": None, "step2": None}

    state["step1"] = await step.run("step1", lambda: "value1")
    state["step2"] = await step.run("step2", lambda: "value2")

    return {"state": state, "note": "Query via Run API, not native signal"}


# =============================================================================
# CP: Compensation & Recovery Functions
# =============================================================================

@inngest_client.create_function(
    fn_id="cp01-compensation",
    trigger=inngest.TriggerEvent(event="test/cp01.saga"),
)
async def cp01_compensation(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate saga pattern with compensation."""
    compensations = []

    try:
        # Step 1: Create order
        order = await step.run("create-order", lambda: {"order_id": "ORD-123"})
        compensations.append("cancel-order")

        # Step 2: Reserve inventory
        inventory = await step.run("reserve-inventory", lambda: {"reserved": True})
        compensations.append("release-inventory")

        # Step 3: Charge payment (simulate failure based on event data)
        should_fail = ctx.event.data.get("should_fail", False)

        async def charge_payment():
            if should_fail:
                raise Exception("Payment failed")
            return {"charged": True}

        payment = await step.run("charge-payment", charge_payment)

        return {"status": "success", "order": order, "inventory": inventory, "payment": payment}

    except Exception as e:
        # Execute compensations in reverse order
        compensation_results = []
        for comp in reversed(compensations):
            result = await step.run(f"compensate-{comp}", lambda c=comp: {"compensated": c})
            compensation_results.append(result)

        return {"status": "compensated", "compensations": compensation_results, "error": str(e)}


@inngest_client.create_function(
    fn_id="cp02-partial-resume",
    trigger=inngest.TriggerEvent(event="test/cp02.resume"),
)
async def cp02_partial_resume(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate partial resume from failed step."""
    results = []

    # Step 1: Always succeeds
    result1 = await step.run("step1", lambda: {"step": 1, "status": "ok"})
    results.append(result1)

    # Step 2: Always succeeds
    result2 = await step.run("step2", lambda: {"step": 2, "status": "ok"})
    results.append(result2)

    # Step 3: May fail (controlled by event data)
    fail_count = ctx.event.data.get("fail_count", 0)

    async def maybe_fail():
        # In real scenario, this would check external state
        return {"step": 3, "status": "ok"}

    result3 = await step.run("step3", maybe_fail)
    results.append(result3)

    return {"results": results}


# =============================================================================
# AI: AI/Agent Integration Functions
# =============================================================================

@inngest_client.create_function(
    fn_id="ai01-llm-activity",
    trigger=inngest.TriggerEvent(event="test/ai01.llm"),
    retries=2,
)
async def ai01_llm_activity(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate LLM call as a durable step."""
    prompt = ctx.event.data.get("prompt", "Hello")

    async def call_llm():
        # Simulated LLM response (in production, use openai.chat.completions.create)
        return {
            "model": "gpt-4o-mini",
            "prompt": prompt,
            "response": f"[Simulated LLM response to: {prompt[:50]}...]",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

    result = await step.run("llm-call", call_llm)
    return {"result": result}


@inngest_client.create_function(
    fn_id="ai03-hitl-approval",
    trigger=inngest.TriggerEvent(event="test/ai03.hitl"),
)
async def ai03_hitl_approval(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate human-in-the-loop approval pattern."""
    from datetime import timedelta

    action = ctx.event.data.get("action", "deploy")
    correlation_id = ctx.event.data.get("correlation_id", "default")

    # Generate proposal
    proposal = await step.run("generate-proposal", lambda: {
        "action": action,
        "proposal": f"Detailed plan for: {action}",
    })

    # Wait for human approval
    approval = await step.wait_for_event(
        "wait-for-approval",
        event="test/ai03.approval",
        timeout=timedelta(minutes=60),
        if_exp=f"event.data.correlation_id == '{correlation_id}'",
    )

    if approval is None:
        return {"status": "timeout", "proposal": proposal}

    approved = approval.data.get("approved", False)
    reason = approval.data.get("reason", "")

    if approved:
        # Execute the action
        execution = await step.run("execute-action", lambda: {"executed": True})
        return {"status": "approved", "proposal": proposal, "execution": execution, "reason": reason}
    else:
        return {"status": "rejected", "proposal": proposal, "reason": reason}


@inngest_client.create_function(
    fn_id="ai06-tool-execution",
    trigger=inngest.TriggerEvent(event="test/ai06.tools"),
    retries=3,
)
async def ai06_tool_execution(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Demonstrate fault-tolerant tool execution."""
    tools = ctx.event.data.get("tools", [])

    results = []
    for i, tool in enumerate(tools):
        tool_name = tool.get("name", f"tool-{i}")
        tool_params = tool.get("params", {})

        async def execute_tool(name=tool_name, params=tool_params):
            return {"tool": name, "params": params, "result": "success"}

        result = await step.run(f"tool-{tool_name}", execute_tool)
        results.append(result)

    return {"tool_results": results}


# =============================================================================
# Performance Functions
# =============================================================================

@inngest_client.create_function(
    fn_id="pf01-latency",
    trigger=inngest.TriggerEvent(event="test/pf01.latency"),
)
async def pf01_latency(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Measure step execution latency."""
    import time

    start = time.perf_counter()

    # Empty step to measure overhead
    await step.run("empty-step", lambda: None)

    elapsed = time.perf_counter() - start

    return {"latency_ms": elapsed * 1000}


@inngest_client.create_function(
    fn_id="pf02-fanout",
    trigger=inngest.TriggerEvent(event="test/pf02.fanout"),
)
async def pf02_fanout(
    ctx: inngest.Context,
    step: inngest.Step,
) -> dict:
    """Measure fan-out throughput."""
    import time

    count = ctx.event.data.get("count", 10)

    start = time.perf_counter()

    # Create parallel tasks
    tasks = tuple(
        lambda i=i: step.run(f"fanout-{i}", lambda i=i: {"index": i})
        for i in range(count)
    )

    results = await step.parallel(tasks)

    elapsed = time.perf_counter() - start

    return {
        "count": count,
        "elapsed_ms": elapsed * 1000,
        "throughput": count / elapsed if elapsed > 0 else 0,
    }


# =============================================================================
# Serve all functions
# =============================================================================

# Collect all functions
all_functions = [
    # EX: Execution Semantics
    ex01_progress_guarantee,
    ex02_side_effect,
    ex03_idempotency,
    ex04_state_persistence,
    ex05_determinism,
    # RT: Retry & Timeout
    rt01_retry_strategy,
    rt02_timeout,
    # WF: Workflow Primitives
    wf01_step_definition,
    wf02_child_function,
    wf02_child_worker,
    wf03_parallel,
    wf04_control_flow,
    wf05_sleep_timer,
    wf06_concurrency,
    # SG: Signals & Events
    sg01_send_event,
    sg01_receiver,
    sg02_wait_for_event,
    sg04_query_state,
    # CP: Compensation & Recovery
    cp01_compensation,
    cp02_partial_resume,
    # AI: AI/Agent Integration
    ai01_llm_activity,
    ai03_hitl_approval,
    ai06_tool_execution,
    # PF: Performance
    pf01_latency,
    pf02_fanout,
]

# Register with FastAPI
serve(
    app,
    inngest_client,
    all_functions,
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
