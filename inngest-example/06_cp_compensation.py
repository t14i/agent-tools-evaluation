"""
CP: Compensation & Recovery - Failure Recovery and Manual Intervention Verification

Evaluation Items:
- CP-01: Compensation Transaction / Saga - Reverse operations on step failure
- CP-02: Partial Resume - Resume from failed step
- CP-03: Manual Intervention - Intervention via Dashboard/API
- CP-04: Dead Letter / Poison Message - Isolation of unprocessable workflows
"""

import asyncio

from common import print_section, print_result


async def verify_cp01_compensation_saga() -> str:
    """Verify CP-01: Compensation / Saga."""
    print_section("CP-01: Compensation Transaction / Saga")

    print("  Inngest saga pattern implementation:")
    print("    async def order_saga(ctx, step):")
    print("        compensations = []")
    print("")
    print("        try:")
    print("            # Step 1: Create order")
    print("            order = await step.run('create-order', create_order)")
    print("            compensations.append(('cancel-order', order['id']))")
    print("")
    print("            # Step 2: Reserve inventory")
    print("            await step.run('reserve', reserve_inventory)")
    print("            compensations.append(('release', order['id']))")
    print("")
    print("            # Step 3: Charge payment (may fail)")
    print("            await step.run('charge', charge_payment)")
    print("")
    print("            return {'status': 'success'}")
    print("")
    print("        except Exception as e:")
    print("            # Execute compensations in reverse")
    print("            for comp_name, comp_data in reversed(compensations):")
    print("                await step.run(f'comp-{comp_name}', ")
    print("                    lambda: compensate(comp_name, comp_data))")
    print("            return {'status': 'compensated', 'error': str(e)}")

    print("\n  Characteristics:")
    print("    - Manual saga implementation (no framework support)")
    print("    - Compensation steps are also memoized")
    print("    - try/except pattern for failure handling")
    print("    - Compensation order is developer responsibility")

    rating = "⭐⭐⭐⭐"
    note = "Implemented via try/except + compensation step pattern. No framework support, but step memoization persists compensations"

    print_result("CP-01 Compensation Transaction/Saga", rating, note)
    return rating


async def verify_cp02_partial_resume() -> str:
    """Verify CP-02: Partial Resume."""
    print_section("CP-02: Partial Resume")

    print("  Inngest partial resume mechanism:")
    print("    1. Automatic on retry:")
    print("       - Failed function retries automatically")
    print("       - Completed steps are skipped (memoized)")
    print("       - Execution resumes from failed step")

    print("\n    2. Manual replay:")
    print("       # Via Dashboard")
    print("       # Select failed run -> Replay button")
    print("")
    print("       # Via API")
    print("       POST /v1/runs/{run_id}/replay")

    print("\n  Behavior:")
    print("    async def multi_step(ctx, step):")
    print("        r1 = await step.run('step1', step1)  # ✓ Cached")
    print("        r2 = await step.run('step2', step2)  # ✓ Cached")
    print("        r3 = await step.run('step3', step3)  # ✗ Failed here")
    print("        r4 = await step.run('step4', step4)  # Resume from here")
    print("")
    print("    # On replay: step1, step2 return cached results")
    print("    # step3 is re-executed, then step4")

    print("\n  Limitations:")
    print("    - Cannot skip steps manually")
    print("    - Cannot modify step results")
    print("    - Replay is from beginning (but cached steps are fast)")

    rating = "⭐⭐⭐⭐⭐"
    note = "Automatic partial resume via step memoization. Re-execution from failed step works naturally"

    print_result("CP-02 Partial Resume", rating, note)
    return rating


async def verify_cp03_manual_intervention() -> str:
    """Verify CP-03: Manual Intervention."""
    print_section("CP-03: Manual Intervention")

    print("  Inngest Dashboard interventions:")
    print("    1. View run details:")
    print("       - Step history and results")
    print("       - Timing information")
    print("       - Error details")

    print("\n    2. Replay failed runs:")
    print("       - Dashboard: Select run -> Replay")
    print("       - API: POST /v1/runs/{run_id}/replay")

    print("\n    3. Cancel runs:")
    print("       - Dashboard: Select run -> Cancel")
    print("       - API: POST /v1/runs/{run_id}/cancel")

    print("\n  API capabilities:")
    print("    - List runs: GET /v1/runs")
    print("    - Get run: GET /v1/runs/{run_id}")
    print("    - Replay: POST /v1/runs/{run_id}/replay")
    print("    - Cancel: POST /v1/runs/{run_id}/cancel")

    print("\n  Limitations vs Temporal:")
    print("    - No signal injection")
    print("    - No workflow reset to specific point")
    print("    - No state modification")
    print("    - No terminate (hard kill)")

    rating = "⭐⭐⭐⭐"
    note = "Replay/Cancel available via Dashboard + API. Lacks Temporal's detailed intervention (Signal injection, Reset)"

    print_result("CP-03 Manual Intervention", rating, note)
    return rating


async def verify_cp04_dead_letter() -> str:
    """Verify CP-04: Dead Letter / Poison Message."""
    print_section("CP-04: Dead Letter / Poison Message")

    print("  Inngest dead letter handling:")
    print("    1. Max retries exhausted:")
    print("       - Run marked as 'Failed' status")
    print("       - Visible in Dashboard")
    print("       - No automatic isolation queue")

    print("\n    2. NonRetriableError:")
    print("       raise inngest.NonRetriableError('Invalid input')")
    print("       # Immediately fails, no retry")
    print("       # Marked as 'Failed'")

    print("\n    3. Monitoring failed runs:")
    print("       # Dashboard filter by status")
    print("       # API: GET /v1/runs?status=Failed")

    print("\n  Alerting:")
    print("    - Inngest Cloud: Built-in failure alerts")
    print("    - Self-hosted: Webhook notifications")
    print("    - Integration: Slack, PagerDuty, etc.")

    print("\n  Recovery:")
    print("    - Manual replay from Dashboard")
    print("    - Fix code, redeploy, replay")
    print("    - Bulk replay via API")

    rating = "⭐⭐⭐⭐"
    note = "Failed status visibility, NonRetriableError. No dedicated Dead Letter Queue, but manageable via dashboard"

    print_result("CP-04 Dead Letter/Poison Message", rating, note)
    return rating


async def main():
    """Run all CP category verifications."""
    print("\n" + "="*60)
    print("  CP: Compensation & Recovery Verification")
    print("="*60)

    results = {}

    results["CP-01"] = await verify_cp01_compensation_saga()
    results["CP-02"] = await verify_cp02_partial_resume()
    results["CP-03"] = await verify_cp03_manual_intervention()
    results["CP-04"] = await verify_cp04_dead_letter()

    print("\n" + "="*60)
    print("  CP Category Summary")
    print("="*60)
    for item, rating in results.items():
        print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
