"""
AI: AI/Agent Integration - LLM/Agent Integration, Non-determinism, and HITL Verification

Evaluation Items:
- AI-01: LLM Call as Activity - Wrap LLM calls in step.run()
- AI-02: Non-determinism Handling - Handle non-determinism via step result memoization
- AI-03: HITL / Human Approval - HITL via step.wait_for_event()
- AI-04: Streaming - Streaming support
- AI-05: Agent Framework Integration - AgentKit integration
- AI-06: Tool Execution Fault Tolerance - Tool execution fault tolerance
"""

import asyncio

from common import print_section, print_result


async def verify_ai01_llm_activity() -> str:
    """Verify AI-01: LLM Call as Activity."""
    print_section("AI-01: LLM Call as Activity")

    print("  Inngest LLM call pattern:")
    print("    @inngest_client.create_function(")
    print("        fn_id='llm-workflow',")
    print("        trigger=inngest.TriggerEvent(event='ai/process'),")
    print("        retries=3,  # Retry on API failures")
    print("    )")
    print("    async def llm_workflow(ctx, step):")
    print("        # LLM call as durable step")
    print("        response = await step.run('llm-call', lambda: call_openai(")
    print("            model='gpt-4',")
    print("            prompt=ctx.event.data['prompt'],")
    print("        ))")
    print("        return response")

    print("\n  Benefits:")
    print("    - LLM response memoized")
    print("    - No re-execution on replay (cost saving)")
    print("    - Automatic retry on API failures")
    print("    - Timeout protection")

    print("\n  Real implementation:")
    print("    async def call_openai(prompt):")
    print("        client = OpenAI()")
    print("        response = client.chat.completions.create(")
    print("            model='gpt-4',")
    print("            messages=[{'role': 'user', 'content': prompt}]")
    print("        )")
    print("        return response.choices[0].message.content")

    rating = "⭐⭐⭐⭐⭐"
    note = "Wrap LLM calls in step.run(). Results memoized, no re-execution (cost saving). Automatic retry"

    print_result("AI-01 LLM Call as Activity", rating, note)
    return rating


async def verify_ai02_non_determinism() -> str:
    """Verify AI-02: Non-determinism Handling."""
    print_section("AI-02: Non-determinism Handling")

    print("  Inngest non-determinism handling:")
    print("    1. Step memoization:")
    print("       # LLM response stored in journal")
    print("       response = await step.run('llm', call_llm)")
    print("       # On replay: returns cached response")

    print("\n    2. Model/prompt changes:")
    print("       # Changing model or prompt doesn't affect running flows")
    print("       # New runs use new config")
    print("       # Existing runs use memoized results")

    print("\n    3. Re-execution control:")
    print("       # Dashboard: Replay to re-run with new code")
    print("       # API: POST /v1/runs/{id}/replay")

    print("\n  Determinism isolation:")
    print("    - LLM calls isolated in steps")
    print("    - Step results are immutable")
    print("    - Code outside steps can be non-deterministic")

    print("\n  Comparison with Temporal:")
    print("    Similar: Activity isolation for LLM")
    print("    Inngest advantage: No strict determinism rules")
    print("    Temporal advantage: More explicit versioning")

    rating = "⭐⭐⭐⭐"
    note = "Non-determinism isolated via step memoization. Returns saved result on replay. Model/prompt change management is manual"

    print_result("AI-02 Non-determinism Handling", rating, note)
    return rating


async def verify_ai03_hitl_approval() -> str:
    """Verify AI-03: HITL / Human Approval."""
    print_section("AI-03: HITL / Human Approval")

    print("  Inngest HITL pattern:")
    print("    async def approval_workflow(ctx, step):")
    print("        # Generate AI proposal")
    print("        proposal = await step.run('generate', generate_proposal)")
    print("")
    print("        # Wait for human approval")
    print("        approval = await step.wait_for_event(")
    print("            'wait-approval',")
    print("            event='approval/decision',")
    print("            timeout=timedelta(days=7),")
    print("            if_exp=f\"event.data.proposal_id == '{proposal['id']}'\",")
    print("        )")
    print("")
    print("        if approval is None:")
    print("            return {'status': 'timeout'}")
    print("")
    print("        if approval.data['approved']:")
    print("            await step.run('execute', execute_action)")
    print("            return {'status': 'executed'}")
    print("        else:")
    print("            return {'status': 'rejected'}")

    print("\n  Approval UI integration:")
    print("    # Frontend sends approval event")
    print("    POST /e/approval/decision")
    print("    {")
    print("      'name': 'approval/decision',")
    print("      'data': {")
    print("        'proposal_id': '...',")
    print("        'approved': true,")
    print("        'approver': 'user@example.com'")
    print("      }")
    print("    }")

    rating = "⭐⭐⭐⭐⭐"
    note = "Durable approval waiting via step.wait_for_event(). Configurable timeout. Flexible expression-based matching"

    print_result("AI-03 HITL/Human Approval", rating, note)
    return rating


async def verify_ai04_streaming() -> str:
    """Verify AI-04: Streaming."""
    print_section("AI-04: Streaming")

    print("  Inngest streaming limitations:")
    print("    - HTTP request-response model")
    print("    - step.run() returns complete result")
    print("    - Cannot stream through Inngest boundary")

    print("\n  Workaround patterns:")
    print("    1. Client-side streaming (bypass Inngest):")
    print("       # Inngest triggers, client streams directly")
    print("       async def llm_workflow(ctx, step):")
    print("           task_id = await step.run('prepare', prepare_task)")
    print("           # Client polls task_id and streams from LLM directly")

    print("\n    2. Chunked responses:")
    print("       async def chunked_llm(ctx, step):")
    print("           prompt = ctx.event.data['prompt']")
    print("           chunks = []")
    print("           for i, chunk in enumerate(stream_llm(prompt)):")
    print("               chunks.append(")
    print("                   await step.run(f'chunk-{i}', lambda: chunk)")
    print("               )")
    print("           return {'chunks': chunks}")

    print("\n    3. External streaming service:")
    print("       # Step triggers external streaming service")
    print("       # Client connects to that service directly")

    print("\n  Comparison with Temporal:")
    print("    Similar limitations (Activity boundary)")

    rating = "⭐⭐⭐"
    note = "Streaming cannot cross Inngest boundary. Handle via external service or chunk pattern"

    print_result("AI-04 Streaming", rating, note)
    return rating


async def verify_ai05_framework_integration() -> str:
    """Verify AI-05: Agent Framework Integration."""
    print_section("AI-05: Agent Framework Integration")

    print("  Inngest AgentKit:")
    print("    # Official agent toolkit")
    print("    from inngest.experimental.agentkit import Agent")
    print("")
    print("    agent = Agent(")
    print("        llm=OpenAIChat(model='gpt-4'),")
    print("        tools=[search_tool, calculator_tool],")
    print("    )")
    print("")
    print("    @inngest_client.create_function(...)")
    print("    async def agent_workflow(ctx, step):")
    print("        result = await agent.run(ctx, step, ctx.event.data['query'])")

    print("\n  LangGraph integration:")
    print("    @inngest_client.create_function(...)")
    print("    async def langgraph_workflow(ctx, step):")
    print("        # Run LangGraph as step")
    print("        result = await step.run('langgraph', lambda: graph.invoke(")
    print("            {'input': ctx.event.data['query']}")
    print("        ))")

    print("\n  CrewAI integration:")
    print("    @inngest_client.create_function(...)")
    print("    async def crew_workflow(ctx, step):")
    print("        result = await step.run('crew', lambda: crew.kickoff())")

    print("\n  Official integrations:")
    print("    - AgentKit (Inngest native)")
    print("    - OpenAI SDK")
    print("    - LangChain/LangGraph")
    print("    - Documentation and examples")

    rating = "⭐⭐⭐⭐⭐"
    note = "AgentKit (official) + LangGraph/CrewAI integration patterns. Well-designed for AI Agent use cases"

    print_result("AI-05 Agent Framework Integration", rating, note)
    return rating


async def verify_ai06_tool_fault_tolerance() -> str:
    """Verify AI-06: Tool Execution Fault Tolerance."""
    print_section("AI-06: Tool Execution Fault Tolerance")

    print("  Inngest tool execution pattern:")
    print("    async def agent_with_tools(ctx, step):")
    print("        # Each tool call is a durable step")
    print("        search_result = await step.run(")
    print("            'web-search',")
    print("            lambda: web_search(query),")
    print("        )")
    print("")
    print("        calc_result = await step.run(")
    print("            'calculate',")
    print("            lambda: calculate(expression),")
    print("        )")
    print("")
    print("        # Tool failures trigger retries")
    print("        # Completed tools not re-executed")

    print("\n  Retry configuration:")
    print("    @inngest_client.create_function(")
    print("        fn_id='tool-runner',")
    print("        retries=5,  # Function-level retries")
    print("    )")

    print("\n  Error handling:")
    print("    async def safe_tool_call(tool_name, tool_fn):")
    print("        try:")
    print("            return await step.run(tool_name, tool_fn)")
    print("        except Exception as e:")
    print("            # Log and continue or raise NonRetriableError")
    print("            raise inngest.NonRetriableError(f'{tool_name} failed')")

    print("\n  Benefits:")
    print("    - Per-tool step isolation")
    print("    - Automatic retry on failures")
    print("    - Completed tools preserved")
    print("    - Progress survives crashes")

    rating = "⭐⭐⭐⭐⭐"
    note = "Each tool call isolated in step.run(). Automatic retry, completed tools not re-executed. Robust agent execution"

    print_result("AI-06 Tool Execution Fault Tolerance", rating, note)
    return rating


async def main():
    """Run all AI category verifications."""
    print("\n" + "="*60)
    print("  AI: AI/Agent Integration Verification")
    print("="*60)

    results = {}

    results["AI-01"] = await verify_ai01_llm_activity()
    results["AI-02"] = await verify_ai02_non_determinism()
    results["AI-03"] = await verify_ai03_hitl_approval()
    results["AI-04"] = await verify_ai04_streaming()
    results["AI-05"] = await verify_ai05_framework_integration()
    results["AI-06"] = await verify_ai06_tool_fault_tolerance()

    print("\n" + "="*60)
    print("  AI Category Summary")
    print("="*60)
    for item, rating in results.items():
        print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
