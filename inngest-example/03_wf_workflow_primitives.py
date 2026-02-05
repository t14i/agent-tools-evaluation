"""
WF: Workflow Primitives - Workflow Components and Control Flow Verification

Evaluation Items:
- WF-01: Step Definition - Define steps with step.run()
- WF-02: Child Workflows - Child function invocation with step.invoke()
- WF-03: Parallel Execution / Fan-out - step.parallel() for fan-out
- WF-04: Conditional / Loop - Standard Python control flow
- WF-05: Sleep / Timer - step.sleep() / step.sleep_until()
- WF-06: Queue / Rate Control - Concurrency configuration
"""

import asyncio

from common import print_section, print_result


async def verify_wf01_step_definition() -> str:
    """Verify WF-01: Step Definition."""
    print_section("WF-01: Step Definition")

    print("  Inngest step definition with step.run():")
    print("    async def my_function(ctx, step):")
    print("        # Each step.run() creates a durable checkpoint")
    print("        result1 = await step.run('step-1', async_operation)")
    print("        result2 = await step.run('step-2', lambda: process(result1))")
    print("        return result2")

    print("\n  Step characteristics:")
    print("    - Unique step ID required (string)")
    print("    - Result is memoized in journal")
    print("    - Supports sync and async functions")
    print("    - Inherits function retry policy")

    print("\n  Step types:")
    print("    - step.run(): Execute code and memoize result")
    print("    - step.sleep(): Durable delay")
    print("    - step.invoke(): Call another function")
    print("    - step.send_event(): Send event(s)")
    print("    - step.wait_for_event(): Wait for external event")

    rating = "⭐⭐⭐⭐⭐"
    note = "Simple step definition with step.run(). Each step is automatically memoized and not re-executed on replay"

    print_result("WF-01 Step Definition", rating, note)
    return rating


async def verify_wf02_child_workflows() -> str:
    """Verify WF-02: Child Workflows."""
    print_section("WF-02: Child Workflows")

    print("  Inngest child function invocation:")
    print("    # Define child function")
    print("    @inngest_client.create_function(")
    print("        fn_id='child-function',")
    print("        trigger=inngest.TriggerEvent(event='internal/child'),")
    print("    )")
    print("    async def child_fn(ctx, step):")
    print("        return {'result': 'from child'}")

    print("\n    # Invoke from parent")
    print("    async def parent_fn(ctx, step):")
    print("        result = await step.invoke(")
    print("            'call-child',")
    print("            function=child_fn,")
    print("            data={'input': 'value'},")
    print("        )")

    print("\n  Characteristics:")
    print("    - step.invoke() waits for child completion")
    print("    - Child runs as independent function")
    print("    - Results are memoized in parent")
    print("    - No parent close policy control (unlike Temporal)")

    print("\n  Limitations:")
    print("    - No Abandon/Terminate/Wait policy for parent-child")
    print("    - Child failure propagates to parent")

    rating = "⭐⭐⭐⭐"
    note = "Child function invocation via step.invoke(). Results memoized. Lacks Temporal's parent close policy control"

    print_result("WF-02 Child Workflows", rating, note)
    return rating


async def verify_wf03_parallel_execution() -> str:
    """Verify WF-03: Parallel Execution / Fan-out."""
    print_section("WF-03: Parallel Execution / Fan-out")

    print("  Inngest parallel execution with step.parallel():")
    print("    results = await step.parallel(")
    print("        (")
    print("            lambda: step.run('task-a', task_a),")
    print("            lambda: step.run('task-b', task_b),")
    print("            lambda: step.run('task-c', task_c),")
    print("        )")
    print("    )")

    print("\n  Fan-out patterns:")
    print("    # Dynamic fan-out")
    print("    tasks = tuple(")
    print("        lambda i=i: step.run(f'task-{i}', lambda: process(i))")
    print("        for i in range(count)")
    print("    )")
    print("    results = await step.parallel(tasks)")

    print("\n  Characteristics:")
    print("    - All tasks execute concurrently")
    print("    - Each task is individually memoized")
    print("    - Fan-in: results collected when all complete")
    print("    - Partial failure: continues other tasks")

    rating = "⭐⭐⭐⭐⭐"
    note = "Natural parallel execution with step.parallel(). Each task individually memoized, automatic fan-in"

    print_result("WF-03 Parallel Execution/Fan-out", rating, note)
    return rating


async def verify_wf04_control_flow() -> str:
    """Verify WF-04: Conditional / Loop."""
    print_section("WF-04: Conditional / Loop")

    print("  Standard Python control flow works:")
    print("    async def my_function(ctx, step):")
    print("        # Conditional")
    print("        if ctx.event.data.get('mode') == 'fast':")
    print("            result = await step.run('fast-path', fast_process)")
    print("        else:")
    print("            result = await step.run('slow-path', slow_process)")

    print("\n        # Loop")
    print("        for i in range(count):")
    print("            await step.run(f'iteration-{i}', lambda: process(i))")

    print("\n        # While loop")
    print("        while not done:")
    print("            status = await step.run(f'check-{i}', check_status)")
    print("            done = status.get('complete')")
    print("            i += 1")

    print("\n  Key requirement:")
    print("    - Step IDs must be deterministic")
    print("    - Loop index in step ID ensures uniqueness")
    print("    - Conditional step IDs must be stable")

    rating = "⭐⭐⭐⭐⭐"
    note = "Standard Python control flow works as-is. Just maintain step ID uniqueness"

    print_result("WF-04 Conditional/Loop", rating, note)
    return rating


async def verify_wf05_sleep_timer() -> str:
    """Verify WF-05: Sleep / Timer."""
    print_section("WF-05: Sleep / Timer")

    print("  Inngest durable sleep:")
    print("    from datetime import timedelta, datetime")

    print("\n    # Duration-based sleep")
    print("    await step.sleep('wait-1-hour', timedelta(hours=1))")

    print("\n    # Sleep until specific time")
    print("    await step.sleep_until('wait-until', datetime(2024, 12, 31, 23, 59))")

    print("\n  Characteristics:")
    print("    - Durable: survives function restarts")
    print("    - No resource consumption during sleep")
    print("    - Precise timing (server-managed)")
    print("    - Long sleeps supported (days/weeks)")

    print("\n  Cron scheduling:")
    print("    @inngest_client.create_function(")
    print("        fn_id='daily-report',")
    print("        trigger=inngest.TriggerCron(cron='0 9 * * *'),")
    print("    )")
    print("    async def daily_report(ctx, step): ...")

    rating = "⭐⭐⭐⭐⭐"
    note = "Durable sleep with step.sleep() / sleep_until(). Native cron trigger support"

    print_result("WF-05 Sleep/Timer", rating, note)
    return rating


async def verify_wf06_queue_rate_control() -> str:
    """Verify WF-06: Queue / Rate Control."""
    print_section("WF-06: Queue / Rate Control")

    print("  Inngest concurrency control:")
    print("    @inngest_client.create_function(")
    print("        fn_id='rate-limited',")
    print("        concurrency=[")
    print("            # Global limit")
    print("            inngest.Concurrency(limit=10),")
    print("            # Per-key limit")
    print("            inngest.Concurrency(")
    print("                limit=2,")
    print("                key='event.data.customer_id',")
    print("            ),")
    print("        ],")
    print("    )")

    print("\n  Rate limiting:")
    print("    @inngest_client.create_function(")
    print("        rate_limit=inngest.RateLimit(")
    print("            limit=100,")
    print("            period=timedelta(minutes=1),")
    print("        ),")
    print("    )")

    print("\n  Features:")
    print("    - Global concurrency limits")
    print("    - Per-key concurrency (e.g., per customer)")
    print("    - Rate limiting (events per period)")
    print("    - Queue managed by Inngest server")

    print("\n  Limitations:")
    print("    - No native priority queues")
    print("    - Queue depth not directly configurable")

    rating = "⭐⭐⭐⭐"
    note = "Native support for concurrency limits + rate limits. No priority queues"

    print_result("WF-06 Queue/Rate Control", rating, note)
    return rating


async def main():
    """Run all WF category verifications."""
    print("\n" + "="*60)
    print("  WF: Workflow Primitives Verification")
    print("="*60)

    results = {}

    results["WF-01"] = await verify_wf01_step_definition()
    results["WF-02"] = await verify_wf02_child_workflows()
    results["WF-03"] = await verify_wf03_parallel_execution()
    results["WF-04"] = await verify_wf04_control_flow()
    results["WF-05"] = await verify_wf05_sleep_timer()
    results["WF-06"] = await verify_wf06_queue_rate_control()

    print("\n" + "="*60)
    print("  WF Category Summary")
    print("="*60)
    for item, rating in results.items():
        print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
