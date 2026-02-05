"""
VR: Versioning & Migration - Code Changes and Schema Changes Verification

Evaluation Items:
- VR-01: Workflow Versioning - Versioning via function_id
- VR-02: Breaking Change Detection - How breaking changes are detected
- VR-03: Migration Strategy - Migrating from old to new version
- VR-04: Schema Evolution - Resilience to input/output schema changes
"""

import asyncio

from common import print_section, print_result


async def verify_vr01_workflow_versioning() -> str:
    """Verify VR-01: Workflow Versioning."""
    print_section("VR-01: Workflow Versioning")

    print("  Inngest versioning approaches:")
    print("    1. Function ID versioning:")
    print("       @inngest_client.create_function(")
    print("           fn_id='order-processor-v2',  # New version")
    print("           trigger=inngest.TriggerEvent(event='order/created'),")
    print("       )")

    print("\n    2. Deployment-based versioning:")
    print("       # Deploy new code, existing runs continue on old code")
    print("       # New events go to new code automatically")

    print("\n  Key characteristics:")
    print("    - No automatic version routing (unlike Temporal Build ID)")
    print("    - In-flight runs use the code at invocation time")
    print("    - Step results are immutable in journal")
    print("    - New steps can be added, existing steps can't change")

    print("\n  Safe changes:")
    print("    - Adding new steps after existing ones")
    print("    - Changing code outside steps")
    print("    - Adding new functions")

    print("\n  Breaking changes:")
    print("    - Changing step IDs")
    print("    - Removing steps that haven't completed")
    print("    - Changing step order")

    rating = "⭐⭐⭐"
    note = "Versioning via function_id. Not as sophisticated as Temporal's Build ID routing or patching API, but basic support available"

    print_result("VR-01 Workflow Versioning", rating, note)
    return rating


async def verify_vr02_breaking_change_detection() -> str:
    """Verify VR-02: Breaking Change Detection."""
    print_section("VR-02: Breaking Change Detection")

    print("  Inngest breaking change detection:")
    print("    - No automatic detection during deployment")
    print("    - Step ID mismatch causes runtime error")
    print("    - Dashboard shows failed runs")

    print("\n  Manual detection methods:")
    print("    1. Review step IDs before deployment")
    print("    2. Test with production-like data")
    print("    3. Monitor dashboard after deployment")

    print("\n  Comparison with Temporal:")
    print("    Temporal: NonDeterministicWorkflowError on replay")
    print("    Temporal: Replay tests in CI/CD")
    print("    Inngest: No equivalent automated detection")

    print("\n  Best practices:")
    print("    - Use stable, meaningful step IDs")
    print("    - Version function IDs for breaking changes")
    print("    - Drain existing runs before major changes")
    print("    - Test replay scenarios manually")

    rating = "⭐⭐"
    note = "No automatic breaking change detection. Step ID mismatch causes runtime error. Lacks Temporal's replay test safety net"

    print_result("VR-02 Breaking Change Detection", rating, note)
    return rating


async def verify_vr03_migration_strategy() -> str:
    """Verify VR-03: Migration Strategy."""
    print_section("VR-03: Migration Strategy")

    print("  Inngest migration approaches:")
    print("    1. Blue-green deployment:")
    print("       - Deploy new function with new fn_id")
    print("       - Route new events to new function")
    print("       - Wait for old runs to complete")
    print("       - Remove old function")

    print("\n    2. Gradual rollout:")
    print("       - Use event data to route versions")
    print("       @inngest_client.create_function(")
    print("           fn_id='processor-v2',")
    print("           trigger=inngest.TriggerEvent(")
    print("               event='order/created',")
    print("               expression='event.data.version == 2',")
    print("           ),")
    print("       )")

    print("\n    3. In-place update (safe changes only):")
    print("       - Add new steps after existing ones")
    print("       - Don't change completed step IDs")
    print("       - Existing runs will pick up new steps")

    print("\n  Drain/cutover:")
    print("    - No automatic drain mechanism")
    print("    - Monitor dashboard for in-flight runs")
    print("    - Manually wait for completion")

    rating = "⭐⭐⭐"
    note = "Blue-green/gradual rollout possible but no automatic drain feature. Manual migration management required"

    print_result("VR-03 Migration Strategy", rating, note)
    return rating


async def verify_vr04_schema_evolution() -> str:
    """Verify VR-04: Schema Evolution."""
    print_section("VR-04: Schema Evolution")

    print("  Inngest schema evolution:")
    print("    # Event data is JSON - flexible schema")
    print("    async def my_function(ctx, step):")
    print("        # Safe: Optional field with default")
    print("        new_field = ctx.event.data.get('new_field', 'default')")

    print("\n        # Safe: Type coercion")
    print("        count = int(ctx.event.data.get('count', 0))")

    print("\n  Pydantic validation pattern:")
    print("    from pydantic import BaseModel, Field")
    print("")
    print("    class OrderEventV2(BaseModel):")
    print("        order_id: str")
    print("        amount: float")
    print("        currency: str = 'USD'  # New field with default")
    print("")
    print("    async def process_order(ctx, step):")
    print("        order = OrderEventV2(**ctx.event.data)")

    print("\n  Best practices:")
    print("    - Use optional fields with defaults")
    print("    - Validate early with Pydantic")
    print("    - Version event names for major changes")
    print("    - Document schema changes")

    rating = "⭐⭐⭐⭐"
    note = "Flexible schema evolution with JSON event data. Pydantic for validation. Version event names for major changes"

    print_result("VR-04 Schema Evolution", rating, note)
    return rating


async def main():
    """Run all VR category verifications."""
    print("\n" + "="*60)
    print("  VR: Versioning & Migration Verification")
    print("="*60)

    results = {}

    results["VR-01"] = await verify_vr01_workflow_versioning()
    results["VR-02"] = await verify_vr02_breaking_change_detection()
    results["VR-03"] = await verify_vr03_migration_strategy()
    results["VR-04"] = await verify_vr04_schema_evolution()

    print("\n" + "="*60)
    print("  VR Category Summary")
    print("="*60)
    for item, rating in results.items():
        print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
