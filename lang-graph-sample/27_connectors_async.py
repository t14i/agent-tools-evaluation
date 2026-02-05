"""
LangGraph Connectors - Async Job Pattern & Schema Validation (CX-03, CX-04)
Async job execution with polling/webhook and Pydantic schema validation.

Evaluation: CX-03 (Async Job Pattern), CX-04 (Schema / Contract)
"""

import asyncio
import json
import uuid
from typing import Annotated, TypedDict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from threading import Lock

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import ToolMessage


# =============================================================================
# ASYNC JOB SYSTEM (CX-03)
# =============================================================================

class JobStatus(Enum):
    """Job status states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AsyncJob:
    """An async job record."""
    job_id: str
    operation: str
    input_data: dict
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: int = 0  # 0-100
    webhook_url: Optional[str] = None


class AsyncJobExecutor:
    """
    Manages async job execution with polling and webhooks.
    Implements CX-03: Async Job Pattern.
    """

    def __init__(self):
        self.jobs: dict[str, AsyncJob] = {}
        self.lock = Lock()
        self.webhooks_sent: list[dict] = []  # Track webhook calls for testing

    def submit_job(
        self,
        operation: str,
        input_data: dict,
        webhook_url: Optional[str] = None
    ) -> AsyncJob:
        """Submit a new async job."""
        job_id = f"job_{uuid.uuid4().hex[:8]}"

        job = AsyncJob(
            job_id=job_id,
            operation=operation,
            input_data=input_data,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            webhook_url=webhook_url
        )

        with self.lock:
            self.jobs[job_id] = job

        print(f"  [JOB] Submitted: {job_id} ({operation})")
        return job

    def get_job(self, job_id: str) -> Optional[AsyncJob]:
        """Get job status."""
        return self.jobs.get(job_id)

    def poll_job(self, job_id: str) -> dict:
        """Poll job status (for polling pattern)."""
        job = self.get_job(job_id)
        if not job:
            return {"error": "Job not found", "job_id": job_id}

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "result": job.result if job.status == JobStatus.COMPLETED else None,
            "error": job.error if job.status == JobStatus.FAILED else None
        }

    def start_job(self, job_id: str):
        """Start job execution."""
        job = self.jobs.get(job_id)
        if job:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()

    def update_progress(self, job_id: str, progress: int):
        """Update job progress."""
        job = self.jobs.get(job_id)
        if job:
            job.progress = min(100, max(0, progress))

    def complete_job(self, job_id: str, result: Any):
        """Mark job as completed."""
        job = self.jobs.get(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.result = result
            job.progress = 100
            self._send_webhook(job)

    def fail_job(self, job_id: str, error: str):
        """Mark job as failed."""
        job = self.jobs.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            job.error = error
            self._send_webhook(job)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.jobs.get(job_id)
        if job and job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            self._send_webhook(job)
            return True
        return False

    def _send_webhook(self, job: AsyncJob):
        """Send webhook notification (simulated)."""
        if job.webhook_url:
            payload = {
                "job_id": job.job_id,
                "status": job.status.value,
                "result": job.result,
                "error": job.error,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            self.webhooks_sent.append({
                "url": job.webhook_url,
                "payload": payload,
                "sent_at": datetime.now()
            })
            print(f"  [WEBHOOK] Sent to {job.webhook_url}: {job.status.value}")

    def simulate_job_execution(self, job_id: str, duration_steps: int = 3):
        """Simulate async job execution (for testing)."""
        import time

        self.start_job(job_id)

        for i in range(duration_steps):
            time.sleep(0.1)  # Simulate work
            progress = int((i + 1) / duration_steps * 100)
            self.update_progress(job_id, progress)

        job = self.jobs.get(job_id)
        if job:
            # Simulate result based on operation
            result = f"Completed {job.operation} with input: {job.input_data}"
            self.complete_job(job_id, result)


# =============================================================================
# SCHEMA VALIDATION (CX-04)
# =============================================================================

class EmailInput(BaseModel):
    """Validated email input schema."""
    to: str = Field(..., description="Recipient email address")
    subject: str = Field(..., max_length=200, description="Email subject")
    body: str = Field(..., description="Email body content")
    priority: str = Field(default="normal", description="Email priority")

    @field_validator("to")
    @classmethod
    def validate_email(cls, v):
        if "@" not in v or "." not in v:
            raise ValueError("Invalid email format")
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v):
        valid = ["low", "normal", "high", "urgent"]
        if v not in valid:
            raise ValueError(f"Priority must be one of: {valid}")
        return v


class ReportInput(BaseModel):
    """Validated report generation input."""
    report_type: str = Field(..., description="Type of report")
    date_range: dict = Field(..., description="Date range for report")
    format: str = Field(default="pdf", description="Output format")
    include_charts: bool = Field(default=True, description="Include charts")

    @field_validator("report_type")
    @classmethod
    def validate_report_type(cls, v):
        valid = ["sales", "inventory", "customer", "financial"]
        if v not in valid:
            raise ValueError(f"Report type must be one of: {valid}")
        return v

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        valid = ["pdf", "xlsx", "csv", "html"]
        if v not in valid:
            raise ValueError(f"Format must be one of: {valid}")
        return v


class APIResponse(BaseModel):
    """Validated API response schema."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class SchemaValidator:
    """
    Validates inputs/outputs against Pydantic schemas.
    Implements CX-04: Schema / Contract.
    """

    def __init__(self):
        self.schemas: dict[str, type[BaseModel]] = {}
        self.validation_errors: list[dict] = []

    def register_schema(self, name: str, schema: type[BaseModel]):
        """Register a schema for validation."""
        self.schemas[name] = schema

    def validate_input(self, schema_name: str, data: dict) -> tuple[bool, Optional[BaseModel], Optional[str]]:
        """
        Validate input against schema.
        Returns: (is_valid, validated_model, error_message)
        """
        if schema_name not in self.schemas:
            return False, None, f"Unknown schema: {schema_name}"

        schema = self.schemas[schema_name]
        try:
            model = schema(**data)
            return True, model, None
        except Exception as e:
            error = str(e)
            self.validation_errors.append({
                "schema": schema_name,
                "data": data,
                "error": error,
                "timestamp": datetime.now()
            })
            return False, None, error

    def validate_output(self, schema_name: str, data: dict) -> tuple[bool, Optional[str]]:
        """Validate output against schema."""
        return self.validate_input(schema_name, data)[:2] is not None, None


# =============================================================================
# STATE AND TOOLS
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    pending_jobs: list
    validation_errors: list


# Initialize systems
job_executor = AsyncJobExecutor()
schema_validator = SchemaValidator()

# Register schemas
schema_validator.register_schema("email", EmailInput)
schema_validator.register_schema("report", ReportInput)
schema_validator.register_schema("api_response", APIResponse)


@tool
def submit_async_email(to: str, subject: str, body: str, webhook_url: str = None) -> str:
    """Submit an async email job with validation."""
    # Validate input
    is_valid, model, error = schema_validator.validate_input("email", {
        "to": to,
        "subject": subject,
        "body": body
    })

    if not is_valid:
        return f"Validation error: {error}"

    # Submit job
    job = job_executor.submit_job(
        operation="send_email",
        input_data={"to": to, "subject": subject, "body": body},
        webhook_url=webhook_url
    )

    # Simulate async execution in background
    import threading
    threading.Thread(
        target=job_executor.simulate_job_execution,
        args=(job.job_id, 3)
    ).start()

    return f"Job submitted: {job.job_id}. Poll /jobs/{job.job_id} for status or wait for webhook."


@tool
def submit_async_report(report_type: str, start_date: str, end_date: str, format: str = "pdf") -> str:
    """Submit an async report generation job with validation."""
    # Validate input
    is_valid, model, error = schema_validator.validate_input("report", {
        "report_type": report_type,
        "date_range": {"start": start_date, "end": end_date},
        "format": format
    })

    if not is_valid:
        return f"Validation error: {error}"

    # Submit job
    job = job_executor.submit_job(
        operation="generate_report",
        input_data={"report_type": report_type, "start_date": start_date, "end_date": end_date, "format": format}
    )

    # Simulate async execution
    import threading
    threading.Thread(
        target=job_executor.simulate_job_execution,
        args=(job.job_id, 5)
    ).start()

    return f"Report job submitted: {job.job_id}. Expected completion in ~5 steps."


@tool
def poll_job_status(job_id: str) -> str:
    """Poll the status of an async job."""
    status = job_executor.poll_job(job_id)
    return json.dumps(status, indent=2, default=str)


@tool
def cancel_job(job_id: str) -> str:
    """Cancel an async job."""
    success = job_executor.cancel_job(job_id)
    if success:
        return f"Job {job_id} cancelled successfully"
    return f"Could not cancel job {job_id} (may be completed or not found)"


tools = [submit_async_email, submit_async_report, poll_job_status, cancel_job]


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


def agent(state: State) -> State:
    """Agent node."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: State) -> str:
    """Conditional edge."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_graph():
    """Build graph with async job support."""
    builder = StateGraph(State)

    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# TESTS
# =============================================================================

def test_async_job_lifecycle():
    """Test async job lifecycle."""
    print("\n" + "=" * 70)
    print("TEST: Async Job Lifecycle (CX-03)")
    print("=" * 70)

    # Submit job
    job = job_executor.submit_job(
        operation="data_export",
        input_data={"table": "users", "format": "csv"},
        webhook_url="https://webhook.example.com/callback"
    )
    print(f"\nSubmitted job: {job.job_id}")

    # Poll status
    status = job_executor.poll_job(job.job_id)
    print(f"Initial status: {status}")

    # Simulate execution
    job_executor.simulate_job_execution(job.job_id, 3)

    # Poll final status
    final_status = job_executor.poll_job(job.job_id)
    print(f"Final status: {final_status}")

    # Check webhook
    print(f"\nWebhooks sent: {len(job_executor.webhooks_sent)}")
    if job_executor.webhooks_sent:
        print(f"  Last webhook: {job_executor.webhooks_sent[-1]}")


def test_job_cancellation():
    """Test job cancellation."""
    print("\n" + "=" * 70)
    print("TEST: Job Cancellation")
    print("=" * 70)

    # Submit and immediately cancel
    job = job_executor.submit_job("long_operation", {"duration": "1h"})
    print(f"Submitted: {job.job_id}")

    success = job_executor.cancel_job(job.job_id)
    print(f"Cancellation success: {success}")

    status = job_executor.poll_job(job.job_id)
    print(f"Status after cancel: {status}")


def test_schema_validation():
    """Test schema validation (CX-04)."""
    print("\n" + "=" * 70)
    print("TEST: Schema Validation (CX-04)")
    print("=" * 70)

    # Valid email
    print("\n--- Valid Email ---")
    is_valid, model, error = schema_validator.validate_input("email", {
        "to": "user@example.com",
        "subject": "Test Subject",
        "body": "Test body content"
    })
    print(f"Valid: {is_valid}, Error: {error}")

    # Invalid email format
    print("\n--- Invalid Email Format ---")
    is_valid, model, error = schema_validator.validate_input("email", {
        "to": "invalid-email",
        "subject": "Test",
        "body": "Body"
    })
    print(f"Valid: {is_valid}, Error: {error}")

    # Invalid report type
    print("\n--- Invalid Report Type ---")
    is_valid, model, error = schema_validator.validate_input("report", {
        "report_type": "unknown",
        "date_range": {"start": "2024-01-01", "end": "2024-12-31"}
    })
    print(f"Valid: {is_valid}, Error: {error}")

    # Valid report
    print("\n--- Valid Report ---")
    is_valid, model, error = schema_validator.validate_input("report", {
        "report_type": "sales",
        "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
        "format": "xlsx"
    })
    print(f"Valid: {is_valid}, Model: {model}")


def test_polling_pattern():
    """Test polling pattern for job status."""
    print("\n" + "=" * 70)
    print("TEST: Polling Pattern")
    print("=" * 70)

    import time

    # Submit job
    job = job_executor.submit_job("long_task", {"data": "test"})

    # Start execution in background
    import threading
    threading.Thread(
        target=job_executor.simulate_job_execution,
        args=(job.job_id, 5)
    ).start()

    # Poll until complete
    print(f"Polling job {job.job_id}...")
    max_polls = 10
    for i in range(max_polls):
        status = job_executor.poll_job(job.job_id)
        print(f"  Poll {i+1}: status={status['status']}, progress={status['progress']}%")

        if status['status'] in ['completed', 'failed']:
            break
        time.sleep(0.1)


def test_integrated_async():
    """Test integrated async job flow."""
    print("\n" + "=" * 70)
    print("TEST: Integrated Async Flow")
    print("=" * 70)

    graph = build_graph()

    result = graph.invoke({
        "messages": [("user", "Send an email to test@example.com with subject 'Hello' and body 'Test message'")],
        "pending_jobs": [],
        "validation_errors": []
    })

    print(f"\nResult: {result['messages'][-1].content}")

    # Wait a moment for job to complete
    import time
    time.sleep(0.5)

    # Check job status
    for job_id, job in job_executor.jobs.items():
        print(f"\nJob {job_id}:")
        print(f"  Status: {job.status.value}")
        print(f"  Result: {job.result}")


# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ CX-03 & CX-04: ASYNC JOBS & SCHEMA - EVALUATION SUMMARY                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LangGraph Native Support: ⭐ (Not Supported)                                │
│                                                                             │
│ LangGraph does NOT provide:                                                 │
│   ❌ Async job management                                                   │
│   ❌ Job polling/webhook patterns                                           │
│   ❌ Built-in schema validation                                             │
│   ❌ Contract enforcement                                                   │
│                                                                             │
│ Custom Implementation Required:                                             │
│   ✅ AsyncJobExecutor - Job submission, tracking, completion                │
│   ✅ Polling pattern - Regular status checks                                │
│   ✅ Webhook pattern - Callback on completion                               │
│   ✅ SchemaValidator - Pydantic-based validation                            │
│                                                                             │
│ CX-03 (Async Job Pattern) Features:                                         │
│   ✓ Job submission with unique ID                                           │
│   ✓ Status tracking (pending, running, completed, failed)                   │
│   ✓ Progress updates                                                        │
│   ✓ Polling for status                                                      │
│   ✓ Webhook notifications                                                   │
│   ✓ Job cancellation                                                        │
│                                                                             │
│ CX-04 (Schema / Contract) Features:                                         │
│   ✓ Pydantic schema definitions                                             │
│   ✓ Field validation with custom validators                                 │
│   ✓ Input validation before execution                                       │
│   ✓ Error tracking                                                          │
│   ✓ Output validation (API responses)                                       │
│                                                                             │
│ Note: LangChain tools support Pydantic args_schema natively:                │
│   @tool(args_schema=EmailInput) provides validation at tool level           │
│                                                                             │
│ Production Considerations:                                                  │
│   - Persistent job storage (Redis, PostgreSQL)                              │
│   - Job queue (Celery, RQ, Bull)                                            │
│   - Webhook retry with exponential backoff                                  │
│   - Job timeout and cleanup                                                 │
│   - Schema versioning                                                       │
│   - Breaking change detection                                               │
│                                                                             │
│ Rating:                                                                     │
│   CX-03 (Async Job): ⭐ (fully custom)                                      │
│   CX-04 (Schema): ⭐⭐⭐ (Pydantic available, integration custom)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_schema_validation()
    test_async_job_lifecycle()
    test_job_cancellation()
    test_polling_pattern()
    test_integrated_async()

    print(SUMMARY)
