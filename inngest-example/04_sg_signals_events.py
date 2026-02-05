"""
SG: Signals & Events - External Events, Signals, and Query Verification

Evaluation Items:
- SG-01: External Signals - Send signals with step.send_event()
- SG-02: Wait / Awaitables - Wait for external events with step.wait_for_event()
- SG-03: Event Triggers - Native event-driven
- SG-04: Query - Query API availability check
"""

import asyncio

from common import print_section, print_result


async def verify_sg01_external_signals() -> str:
    """Verify SG-01: External Signals."""
    print_section("SG-01: External Signals")

    print("  Inngest event sending:")
    print("    # Send events to trigger other functions")
    print("    await step.send_event(")
    print("        'send-signal',")
    print("        events=[")
    print("            inngest.Event(")
    print("                name='notification/send',")
    print("                data={'user_id': user_id, 'message': msg},")
    print("            ),")
    print("            inngest.Event(")
    print("                name='audit/log',")
    print("                data={'action': 'notification_sent'},")
    print("            ),")
    print("        ],")
    print("    )")

    print("\n  Characteristics:")
    print("    - Send multiple events in single step")
    print("    - Events are durable (memoized)")
    print("    - Can trigger other functions or external")
    print("    - Fan-out pattern: one event -> many handlers")

    print("\n  External event ingestion:")
    print("    # Via API (REST)")
    print("    POST /e/{event_name}")
    print("    {'name': 'event/name', 'data': {...}}")

    rating = "⭐⭐⭐⭐⭐"
    note = "Event sending via step.send_event(). Bulk event sending supported. Core feature of event-driven architecture"

    print_result("SG-01 External Signals", rating, note)
    return rating


async def verify_sg02_wait_awaitables() -> str:
    """Verify SG-02: Wait / Awaitables."""
    print_section("SG-02: Wait / Awaitables")

    print("  Inngest wait_for_event pattern:")
    print("    # Wait for external event (e.g., human approval)")
    print("    event = await step.wait_for_event(")
    print("        'wait-for-approval',")
    print("        event='approval/granted',")
    print("        timeout=timedelta(hours=24),")
    print("        if_exp=\"event.data.order_id == '\" + order_id + \"'\",")
    print("    )")

    print("\n    if event is None:")
    print("        # Timeout occurred")
    print("        return {'status': 'timeout'}")

    print("\n    # Process approval")
    print("    approved = event.data.get('approved')")

    print("\n  Use cases:")
    print("    - Human approval workflows")
    print("    - Webhook callbacks")
    print("    - External system responses")
    print("    - Multi-step sagas with coordination")

    print("\n  Characteristics:")
    print("    - Durable wait (survives restarts)")
    print("    - Expression-based event matching")
    print("    - Configurable timeout")
    print("    - Returns None on timeout")

    rating = "⭐⭐⭐⭐⭐"
    note = "Durable external event waiting via step.wait_for_event(). Flexible filtering with expression-based matching"

    print_result("SG-02 Wait/Awaitables", rating, note)
    return rating


async def verify_sg03_event_triggers() -> str:
    """Verify SG-03: Event Triggers."""
    print_section("SG-03: Event Triggers")

    print("  Inngest trigger types:")
    print("    # Event trigger (primary pattern)")
    print("    @inngest_client.create_function(")
    print("        fn_id='order-processor',")
    print("        trigger=inngest.TriggerEvent(event='order/created'),")
    print("    )")

    print("\n    # Cron trigger")
    print("    @inngest_client.create_function(")
    print("        fn_id='daily-cleanup',")
    print("        trigger=inngest.TriggerCron(cron='0 0 * * *'),")
    print("    )")

    print("\n    # Multiple triggers")
    print("    @inngest_client.create_function(")
    print("        fn_id='multi-trigger',")
    print("        trigger=[")
    print("            inngest.TriggerEvent(event='order/created'),")
    print("            inngest.TriggerEvent(event='order/updated'),")
    print("        ],")
    print("    )")

    print("\n  Event sources:")
    print("    - REST API: POST /e/{event_name}")
    print("    - SDK: inngest_client.send(event)")
    print("    - step.send_event(): From within functions")
    print("    - Webhooks: Native integration")

    print("\n  Native integrations:")
    print("    - Webhook ingestion")
    print("    - Kafka (via webhook adapter)")
    print("    - Any HTTP source")

    rating = "⭐⭐⭐⭐⭐"
    note = "Native event-driven. Event/Cron triggers, multiple triggers supported. Direct webhook reception"

    print_result("SG-03 Event Triggers", rating, note)
    return rating


async def verify_sg04_query() -> str:
    """Verify SG-04: Query."""
    print_section("SG-04: Query")

    print("  Inngest query capabilities:")
    print("    # No native query handler like Temporal")
    print("    # State inspection via Run API")

    print("\n  Run API:")
    print("    GET /v1/runs/{run_id}")
    print("    # Returns run status, output, step history")

    print("\n  Workarounds for state query:")
    print("    1. External state store:")
    print("       await step.run('update-state', lambda: db.update(state))")
    print("       # Query DB directly for current state")

    print("\n    2. Event-based status:")
    print("       await step.send_event('status-update', events=[")
    print("           inngest.Event(name='status/updated', data=state)")
    print("       ])")
    print("       # Subscribe to status events")

    print("\n  Comparison with Temporal:")
    print("    Temporal: @workflow.query for synchronous state read")
    print("    Inngest: No equivalent - use Run API or external store")

    print("\n  Limitations:")
    print("    - Cannot query running function's internal state")
    print("    - Only completed step results visible")
    print("    - Real-time state requires external mechanism")

    rating = "⭐⭐⭐"
    note = "No @workflow.query like Temporal. Step results viewable via Run API, but internal state query during execution not possible"

    print_result("SG-04 Query", rating, note)
    return rating


async def main():
    """Run all SG category verifications."""
    print("\n" + "="*60)
    print("  SG: Signals & Events Verification")
    print("="*60)

    results = {}

    results["SG-01"] = await verify_sg01_external_signals()
    results["SG-02"] = await verify_sg02_wait_awaitables()
    results["SG-03"] = await verify_sg03_event_triggers()
    results["SG-04"] = await verify_sg04_query()

    print("\n" + "="*60)
    print("  SG Category Summary")
    print("="*60)
    for item, rating in results.items():
        print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
