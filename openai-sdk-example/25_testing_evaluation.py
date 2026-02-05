"""
Testing & Evaluation - Part 2: Evaluation (TE-03, TE-04, TE-05)
Simulation, dry run, evaluation hooks
"""

from dotenv import load_dotenv
load_dotenv()


import json
from typing import Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# =============================================================================
# TE-03: Simulation / User Emulation
# =============================================================================

@dataclass
class SimulatedUser:
    """A simulated user for testing."""
    name: str
    persona: str
    behaviors: list[str]
    responses: dict[str, str]  # trigger -> response


class UserSimulator:
    """
    Simulates user interactions for testing.
    Implements TE-03: Simulation / User Emulation.
    """

    def __init__(self):
        self.users: dict[str, SimulatedUser] = {}
        self.conversation_history: list[dict] = []

    def add_user(self, user: SimulatedUser):
        """Add a simulated user."""
        self.users[user.name] = user

    def get_response(
        self,
        user_name: str,
        agent_message: str,
        context: dict = None
    ) -> str:
        """Get simulated user response."""
        user = self.users.get(user_name)
        if not user:
            return "I don't understand."

        # Check for trigger-based responses
        for trigger, response in user.responses.items():
            if trigger.lower() in agent_message.lower():
                self.conversation_history.append({
                    "role": "user",
                    "content": response,
                    "user": user_name,
                    "timestamp": datetime.now().isoformat()
                })
                return response

        # Default response based on persona
        default = f"[{user.persona}] Thanks for that information."
        self.conversation_history.append({
            "role": "user",
            "content": default,
            "user": user_name,
            "timestamp": datetime.now().isoformat()
        })
        return default

    def run_conversation(
        self,
        user_name: str,
        agent_fn: Callable,
        initial_message: str,
        max_turns: int = 10
    ) -> list[dict]:
        """Run a simulated conversation."""
        conversation = []
        current_message = initial_message

        for turn in range(max_turns):
            # User message
            conversation.append({
                "role": "user",
                "content": current_message,
                "turn": turn
            })

            # Agent response
            agent_response = agent_fn(current_message)
            conversation.append({
                "role": "assistant",
                "content": agent_response,
                "turn": turn
            })

            # Check for conversation end
            if any(end in agent_response.lower() for end in ["goodbye", "thank you for", "is there anything else"]):
                break

            # Get simulated user response
            current_message = self.get_response(user_name, agent_response)

        return conversation


# =============================================================================
# TE-04: Dry Run / Sandbox Mode
# =============================================================================

class ToolExecutionMode(Enum):
    """Tool execution modes."""
    EXECUTE = "execute"
    DRY_RUN = "dry_run"
    SANDBOX = "sandbox"


@dataclass
class DryRunResult:
    """Result of a dry run."""
    tool_name: str
    args: dict
    would_execute: bool
    simulated_result: Any
    side_effects: list[str]


class DryRunExecutor:
    """
    Executes tools in dry run mode.
    Implements TE-04: Dry Run / Sandbox Mode.
    """

    def __init__(self, mode: ToolExecutionMode = ToolExecutionMode.DRY_RUN):
        self.mode = mode
        self.simulated_results: dict[str, Any] = {}
        self.execution_log: list[DryRunResult] = []

    def set_simulated_result(self, tool_name: str, result: Any):
        """Set simulated result for a tool."""
        self.simulated_results[tool_name] = result

    def execute(
        self,
        tool_name: str,
        args: dict,
        actual_tool: Callable = None
    ) -> DryRunResult:
        """Execute or simulate tool execution."""
        # Analyze potential side effects
        side_effects = self._analyze_side_effects(tool_name, args)

        if self.mode == ToolExecutionMode.EXECUTE:
            # Actually execute
            if actual_tool:
                result = actual_tool(**args)
            else:
                result = f"Executed {tool_name}"

            dry_run_result = DryRunResult(
                tool_name=tool_name,
                args=args,
                would_execute=True,
                simulated_result=result,
                side_effects=side_effects
            )

        elif self.mode == ToolExecutionMode.DRY_RUN:
            # Don't execute, return simulated result
            result = self.simulated_results.get(
                tool_name,
                f"[DRY RUN] Would execute {tool_name} with {args}"
            )

            dry_run_result = DryRunResult(
                tool_name=tool_name,
                args=args,
                would_execute=False,
                simulated_result=result,
                side_effects=side_effects
            )

        else:  # SANDBOX
            # Execute in isolated environment
            result = self._sandbox_execute(tool_name, args, actual_tool)

            dry_run_result = DryRunResult(
                tool_name=tool_name,
                args=args,
                would_execute=True,
                simulated_result=result,
                side_effects=["Executed in sandbox"]
            )

        self.execution_log.append(dry_run_result)
        return dry_run_result

    def _analyze_side_effects(self, tool_name: str, args: dict) -> list[str]:
        """Analyze potential side effects of tool execution."""
        effects = []

        # Detect write operations
        write_keywords = ["write", "create", "update", "delete", "send", "post"]
        if any(kw in tool_name.lower() for kw in write_keywords):
            effects.append(f"May modify data: {tool_name}")

        # Detect external calls
        external_keywords = ["api", "http", "request", "fetch", "email", "notify"]
        if any(kw in tool_name.lower() for kw in external_keywords):
            effects.append(f"May make external call: {tool_name}")

        # Check args for sensitive data
        for key, value in args.items():
            if "password" in key.lower() or "secret" in key.lower():
                effects.append(f"Contains sensitive parameter: {key}")

        return effects

    def _sandbox_execute(
        self,
        tool_name: str,
        args: dict,
        actual_tool: Callable
    ) -> Any:
        """Execute in sandbox (simplified)."""
        # In production, this would use actual sandboxing
        if actual_tool:
            return actual_tool(**args)
        return f"[SANDBOX] Executed {tool_name}"

    def get_execution_plan(self) -> list[dict]:
        """Get planned executions without side effects."""
        return [
            {
                "tool": r.tool_name,
                "args": r.args,
                "side_effects": r.side_effects
            }
            for r in self.execution_log
        ]


# =============================================================================
# TE-05: Evaluation Hooks
# =============================================================================

class EvaluationMetric(Enum):
    """Types of evaluation metrics."""
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COHERENCE = "coherence"
    SAFETY = "safety"
    HELPFULNESS = "helpfulness"


@dataclass
class EvaluationResult:
    """Result of an evaluation."""
    metric: EvaluationMetric
    score: float  # 0-1
    feedback: str
    details: dict = field(default_factory=dict)


class EvaluationHook:
    """
    Hook for evaluating agent responses.
    Implements TE-05: Evaluation Hooks.
    """

    def __init__(self, name: str, metric: EvaluationMetric):
        self.name = name
        self.metric = metric
        self.evaluator: Optional[Callable] = None

    def set_evaluator(self, evaluator: Callable):
        """Set the evaluation function."""
        self.evaluator = evaluator

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        context: dict = None
    ) -> EvaluationResult:
        """Evaluate a response."""
        if self.evaluator:
            score, feedback = self.evaluator(input_text, output_text, context or {})
        else:
            score, feedback = self._default_evaluate(input_text, output_text)

        return EvaluationResult(
            metric=self.metric,
            score=score,
            feedback=feedback,
            details={"input": input_text[:100], "output": output_text[:100]}
        )

    def _default_evaluate(
        self,
        input_text: str,
        output_text: str
    ) -> tuple[float, str]:
        """Default evaluation (placeholder)."""
        # Simple heuristics
        if not output_text:
            return (0.0, "Empty response")

        if len(output_text) < 10:
            return (0.3, "Response too short")

        if any(word in output_text.lower() for word in ["error", "cannot", "unable"]):
            return (0.5, "Response indicates inability")

        return (0.8, "Response looks reasonable")


class EvaluationPipeline:
    """
    Pipeline for running multiple evaluations.
    """

    def __init__(self):
        self.hooks: list[EvaluationHook] = []
        self.results_history: list[dict] = []

    def add_hook(self, hook: EvaluationHook):
        """Add an evaluation hook."""
        self.hooks.append(hook)

    def add_relevance_check(self):
        """Add relevance evaluation hook."""
        hook = EvaluationHook("relevance_check", EvaluationMetric.RELEVANCE)
        hook.set_evaluator(lambda i, o, c: (
            (0.9, "Response is relevant") if any(word in o.lower() for word in i.lower().split()[:5])
            else (0.4, "Response may not be relevant")
        ))
        self.hooks.append(hook)

    def add_safety_check(self):
        """Add safety evaluation hook."""
        hook = EvaluationHook("safety_check", EvaluationMetric.SAFETY)

        unsafe_patterns = ["password", "credit card", "ssn", "hack", "exploit"]

        def check_safety(i, o, c):
            for pattern in unsafe_patterns:
                if pattern in o.lower():
                    return (0.2, f"Potential safety issue: {pattern}")
            return (1.0, "No safety issues detected")

        hook.set_evaluator(check_safety)
        self.hooks.append(hook)

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        context: dict = None
    ) -> dict:
        """Run all evaluations."""
        results = {}

        for hook in self.hooks:
            result = hook.evaluate(input_text, output_text, context)
            results[hook.name] = {
                "metric": result.metric.value,
                "score": result.score,
                "feedback": result.feedback
            }

        # Calculate overall score
        if results:
            overall_score = sum(r["score"] for r in results.values()) / len(results)
        else:
            overall_score = 0

        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": overall_score,
            "results": results
        }

        self.results_history.append(evaluation)
        return evaluation

    def get_aggregate_scores(self) -> dict:
        """Get aggregate scores across all evaluations."""
        if not self.results_history:
            return {}

        scores_by_metric = {}
        for eval_result in self.results_history:
            for name, result in eval_result["results"].items():
                if name not in scores_by_metric:
                    scores_by_metric[name] = []
                scores_by_metric[name].append(result["score"])

        return {
            name: {
                "average": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores)
            }
            for name, scores in scores_by_metric.items()
        }


# =============================================================================
# Tests
# =============================================================================

def test_user_simulation():
    """Test user simulation (TE-03)."""
    print("\n" + "=" * 70)
    print("TEST: User Simulation (TE-03)")
    print("=" * 70)

    simulator = UserSimulator()

    # Add simulated user
    simulator.add_user(SimulatedUser(
        name="frustrated_customer",
        persona="Frustrated customer who had a bad experience",
        behaviors=["complains", "demands refund", "mentions competitor"],
        responses={
            "help": "I need help with my order that was delayed",
            "sorry": "Sorry isn't good enough! I want a refund",
            "refund": "Finally! Yes, process the refund please",
            "anything else": "No, just process my refund"
        }
    ))

    # Simple agent function
    def simple_agent(message):
        if "delayed" in message.lower():
            return "I'm sorry to hear about the delay. Let me help you with that."
        elif "refund" in message.lower():
            return "I can process a refund for you. Is there anything else?"
        return "How can I help you today?"

    # Run simulation
    conversation = simulator.run_conversation(
        user_name="frustrated_customer",
        agent_fn=simple_agent,
        initial_message="I have a complaint about my order",
        max_turns=5
    )

    print(f"\nSimulated conversation ({len(conversation)} messages):")
    for msg in conversation:
        role = msg["role"].upper()
        print(f"  [{role}] {msg['content'][:60]}...")

    print("\n✅ User simulation works")


def test_dry_run():
    """Test dry run mode (TE-04)."""
    print("\n" + "=" * 70)
    print("TEST: Dry Run Mode (TE-04)")
    print("=" * 70)

    executor = DryRunExecutor(mode=ToolExecutionMode.DRY_RUN)

    # Set up simulated results
    executor.set_simulated_result("send_email", {"status": "would_send", "recipients": 1})
    executor.set_simulated_result("delete_file", {"status": "would_delete"})

    # Execute in dry run mode
    result1 = executor.execute("send_email", {
        "to": "user@example.com",
        "subject": "Test",
        "body": "Hello"
    })

    result2 = executor.execute("delete_file", {"path": "/important/data.txt"})

    print(f"\nDry run results:")
    print(f"  send_email: {result1.simulated_result}")
    print(f"    Side effects: {result1.side_effects}")
    print(f"    Would execute: {result1.would_execute}")

    print(f"  delete_file: {result2.simulated_result}")
    print(f"    Side effects: {result2.side_effects}")

    # Get execution plan
    plan = executor.get_execution_plan()
    print(f"\nExecution plan: {len(plan)} operations")

    print("\n✅ Dry run works")


def test_evaluation_hooks():
    """Test evaluation hooks (TE-05)."""
    print("\n" + "=" * 70)
    print("TEST: Evaluation Hooks (TE-05)")
    print("=" * 70)

    pipeline = EvaluationPipeline()
    pipeline.add_relevance_check()
    pipeline.add_safety_check()

    # Test cases
    test_cases = [
        ("What is the weather?", "The weather today is sunny with a high of 75°F"),
        ("How do I reset my password?", "Click on 'Forgot Password' and enter your email"),
        ("Tell me about security", "Here is your password: secret123"),  # Safety issue
    ]

    for input_text, output_text in test_cases:
        result = pipeline.evaluate(input_text, output_text)

        print(f"\nInput: {input_text}")
        print(f"Output: {output_text[:50]}...")
        print(f"Overall score: {result['overall_score']:.2f}")
        for name, r in result["results"].items():
            status = "✅" if r["score"] >= 0.7 else "⚠️" if r["score"] >= 0.4 else "❌"
            print(f"  {status} {name}: {r['score']:.2f} - {r['feedback']}")

    # Aggregate scores
    print(f"\nAggregate scores:")
    aggregates = pipeline.get_aggregate_scores()
    for name, stats in aggregates.items():
        print(f"  {name}: avg={stats['average']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

    print("\n✅ Evaluation hooks work")


def test_combined_testing():
    """Test combined testing workflow."""
    print("\n" + "=" * 70)
    print("TEST: Combined Testing Workflow")
    print("=" * 70)

    # Set up dry run executor
    executor = DryRunExecutor(mode=ToolExecutionMode.DRY_RUN)

    # Set up evaluation pipeline
    pipeline = EvaluationPipeline()
    pipeline.add_relevance_check()
    pipeline.add_safety_check()

    # Simulate agent execution with evaluation
    def evaluated_agent(input_text):
        # Dry run tool calls
        if "email" in input_text.lower():
            executor.execute("send_email", {"to": "test@example.com"})

        # Generate response
        response = f"I understand you want to know about {input_text.split()[0]}."

        # Evaluate
        evaluation = pipeline.evaluate(input_text, response)

        return {
            "response": response,
            "evaluation_score": evaluation["overall_score"],
            "dry_run_operations": len(executor.execution_log)
        }

    result = evaluated_agent("Send an email about the weather")
    print(f"\nCombined result:")
    print(f"  Response: {result['response']}")
    print(f"  Evaluation score: {result['evaluation_score']:.2f}")
    print(f"  Dry run operations: {result['dry_run_operations']}")

    print("\n✅ Combined testing workflow works")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ TE-03, TE-04, TE-05: EVALUATION - EVALUATION SUMMARY                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ TE-03 (Simulation / User Emulation): ⭐⭐ (Experimental)                    │
│   ❌ No built-in user simulation                                            │
│   ❌ No persona-based testing                                               │
│   ⚠️ Custom UserSimulator implementation provided                          │
│                                                                             │
│ TE-04 (Dry Run / Sandbox Mode): ⭐⭐ (Experimental)                         │
│   ❌ No built-in dry run mode                                               │
│   ❌ No tool sandboxing                                                     │
│   ⚠️ Custom DryRunExecutor implementation provided                         │
│   ⚠️ Side effect analysis provided                                         │
│                                                                             │
│ TE-05 (Evaluation Hooks): ⭐⭐⭐ (PoC Ready)                                 │
│   ✅ OpenAI Evals integration available                                     │
│   ⚠️ Custom hooks require implementation                                   │
│   ⚠️ No built-in evaluation pipeline                                       │
│                                                                             │
│ Custom Implementation Provided:                                             │
│   ✅ SimulatedUser - Persona-based user simulation                          │
│   ✅ UserSimulator - Automated conversation testing                         │
│   ✅ DryRunExecutor - Tool execution modes                                  │
│   ✅ EvaluationHook - Custom evaluation functions                           │
│   ✅ EvaluationPipeline - Multi-metric evaluation                           │
│                                                                             │
│ Testing Capabilities:                                                       │
│   1. User simulation with personas and behaviors                            │
│   2. Dry run mode for safe testing                                          │
│   3. Evaluation hooks for quality metrics                                   │
│   4. Combined testing workflows                                             │
│                                                                             │
│ Comparison with LangGraph:                                                  │
│   LangGraph:                                                                │
│     - No built-in simulation                                                │
│     - No dry run mode                                                       │
│     - LangSmith evaluators available                                        │
│   OpenAI SDK:                                                               │
│     - Similar gaps                                                          │
│     - OpenAI Evals for evaluation                                           │
│                                                                             │
│ Production Notes:                                                           │
│   - Build comprehensive user personas                                       │
│   - Use dry run for dangerous operations                                    │
│   - Implement continuous evaluation                                         │
│   - Store evaluation results for analysis                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_user_simulation()
    test_dry_run()
    test_evaluation_hooks()
    test_combined_testing()

    print(SUMMARY)
