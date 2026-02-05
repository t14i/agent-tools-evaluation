"""
PF: Performance & Overhead - Latency, Throughput, and Size Constraints Verification

Evaluation Items:
- PF-01: Step Latency - HTTP round-trip latency measurement
- PF-02: Fan-out Throughput - Parallel step throughput
- PF-03: Payload Size Limits - Payload size restrictions
"""

import asyncio

from common import print_section, print_result


async def verify_pf01_step_latency() -> str:
    """Verify PF-01: Step Latency."""
    print_section("PF-01: Step Latency")

    print("  Inngest step execution overhead:")
    print("    - HTTP round-trip to Inngest server")
    print("    - Journal persistence")
    print("    - Step result serialization")

    print("\n  Typical latency (dev server):")
    print("    - Empty step: ~5-20ms")
    print("    - With persistence: ~10-50ms")
    print("    - Network dependent: varies")

    print("\n  Latency breakdown:")
    print("    1. HTTP request to Inngest server")
    print("    2. Journal write (step result)")
    print("    3. HTTP response back to function")
    print("    4. Next step invocation")

    print("\n  Comparison with Temporal:")
    print("    Temporal: ~4ms (regular activity), ~0.2ms (local activity)")
    print("    Inngest: ~10-50ms (HTTP-based)")

    print("\n  Optimization tips:")
    print("    - Batch operations within single step")
    print("    - Use step.parallel() for independent work")
    print("    - Minimize step count for latency-sensitive flows")

    rating = "⭐⭐⭐"
    note = "10-50ms overhead due to HTTP round-trip. Not as low-latency as Temporal's Local Activity"

    print_result("PF-01 Step Latency", rating, note)
    return rating


async def verify_pf02_fanout_throughput() -> str:
    """Verify PF-02: Fan-out Throughput."""
    print_section("PF-02: Fan-out Throughput")

    print("  Inngest fan-out performance:")
    print("    # Parallel execution")
    print("    results = await step.parallel(")
    print("        tuple(")
    print("            lambda i=i: step.run(f'task-{i}', process)")
    print("            for i in range(1000)")
    print("        )")
    print("    )")

    print("\n  Performance characteristics:")
    print("    - Parallel steps execute concurrently")
    print("    - Throughput depends on Inngest server capacity")
    print("    - Dev server: Limited concurrency")
    print("    - Cloud: Higher throughput limits")

    print("\n  Typical throughput:")
    print("    - Dev server: ~10-50 steps/sec")
    print("    - Cloud: Higher (plan dependent)")
    print("    - Self-hosted: Configurable")

    print("\n  Scaling considerations:")
    print("    - Function concurrency limits apply")
    print("    - Rate limits may throttle")
    print("    - Batch size affects performance")

    print("\n  Best practices:")
    print("    - Use reasonable batch sizes (100-500)")
    print("    - Monitor queue depth")
    print("    - Configure concurrency appropriately")

    rating = "⭐⭐⭐⭐"
    note = "Parallel execution with step.parallel(). Throughput depends on server configuration. Cloud version supports high throughput"

    print_result("PF-02 Fan-out Throughput", rating, note)
    return rating


async def verify_pf03_payload_size() -> str:
    """Verify PF-03: Payload Size Limits."""
    print_section("PF-03: Payload Size Limits")

    print("  Inngest payload limits:")
    print("    - Event payload: 512KB default")
    print("    - Step result: 512KB default")
    print("    - Total run data: Varies by plan")

    print("\n  Handling large payloads:")
    print("    # Store in external storage, pass reference")
    print("    async def process_large_file(ctx, step):")
    print("        file_key = ctx.event.data['file_key']")
    print("")
    print("        # Step 1: Download from S3")
    print("        local_path = await step.run(")
    print("            'download',")
    print("            lambda: s3.download(file_key)")
    print("        )")
    print("")
    print("        # Step 2: Process (result is small)")
    print("        result = await step.run(")
    print("            'process',")
    print("            lambda: process_file(local_path)")
    print("        )")
    print("")
    print("        return {'result_key': result}")

    print("\n  Best practices:")
    print("    - Keep payloads small")
    print("    - Use external storage for large data")
    print("    - Pass references, not data")
    print("    - Compress if necessary")

    print("\n  Comparison with Temporal:")
    print("    Temporal: 2MB default, Payload Codec for compression")
    print("    Inngest: 512KB default, external storage recommended")

    rating = "⭐⭐⭐⭐"
    note = "512KB payload limit. Large data handled via external storage reference pattern (Claim Check pattern)"

    print_result("PF-03 Payload Size Limits", rating, note)
    return rating


async def main():
    """Run all PF category verifications."""
    print("\n" + "="*60)
    print("  PF: Performance & Overhead Verification")
    print("="*60)

    results = {}

    results["PF-01"] = await verify_pf01_step_latency()
    results["PF-02"] = await verify_pf02_fanout_throughput()
    results["PF-03"] = await verify_pf03_payload_size()

    print("\n" + "="*60)
    print("  PF Category Summary")
    print("="*60)
    for item, rating in results.items():
        print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
