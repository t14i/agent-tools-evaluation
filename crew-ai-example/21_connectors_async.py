"""
21_connectors_async.py - Connectors Async and Schema (CX-03, CX-04)

Purpose: Verify async job pattern and schema/contract
- CX-03: Async Job Pattern - polling/webhook, job tracking
- CX-04: Schema/Contract - input/output validation, versioning

LangGraph Comparison:
- Both require custom implementation for async patterns
- Pydantic integration available in both frameworks
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Type, Optional, Any, Literal
from enum import Enum

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.version import VERSION as PYDANTIC_VERSION


# =============================================================================
# Async Job Pattern (CX-03)
# =============================================================================

class JobStatus(str, Enum):
    """Status of an async job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AsyncJob(BaseModel):
    """Represents an asynchronous job."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    webhook_url: Optional[str] = None
    metadata: dict = {}


class JobStore:
    """
    Storage for async jobs.

    Supports job tracking, status updates, and persistence.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./db/jobs")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, AsyncJob] = {}

    def create(self, operation: str, webhook_url: Optional[str] = None, **metadata) -> AsyncJob:
        """Create a new job."""
        job = AsyncJob(
            operation=operation,
            webhook_url=webhook_url,
            metadata=metadata,
        )
        self.jobs[job.job_id] = job
        self._persist(job)
        print(f"[JobStore] Created job: {job.job_id}")
        return job

    def get(self, job_id: str) -> Optional[AsyncJob]:
        """Get a job by ID."""
        if job_id not in self.jobs:
            self._load(job_id)
        return self.jobs.get(job_id)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: float = 0.0,
        result: Any = None,
        error: str = None,
    ):
        """Update job status."""
        job = self.get(job_id)
        if not job:
            return

        job.status = status
        job.progress = progress

        if status == JobStatus.RUNNING and job.started_at is None:
            job.started_at = datetime.now()

        if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            job.completed_at = datetime.now()

        if result is not None:
            job.result = result

        if error is not None:
            job.error = error

        self._persist(job)
        self._notify_webhook(job)

    def _persist(self, job: AsyncJob):
        """Persist job to storage."""
        file_path = self.storage_path / f"{job.job_id}.json"
        with open(file_path, "w") as f:
            f.write(job.model_dump_json(indent=2))

    def _load(self, job_id: str):
        """Load job from storage."""
        file_path = self.storage_path / f"{job_id}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)
                self.jobs[job_id] = AsyncJob(**data)

    def _notify_webhook(self, job: AsyncJob):
        """Notify webhook of job status change (simulated)."""
        if job.webhook_url:
            print(f"[Webhook] Would notify {job.webhook_url}: {job.status.value}")


class AsyncJobExecutor:
    """
    Executes async jobs with polling support.
    """

    def __init__(self, job_store: JobStore):
        self.job_store = job_store
        self.handlers: dict[str, callable] = {}

    def register_handler(self, operation: str, handler: callable):
        """Register a handler for an operation type."""
        self.handlers[operation] = handler

    async def execute(self, job_id: str):
        """Execute a job asynchronously."""
        job = self.job_store.get(job_id)
        if not job:
            return

        handler = self.handlers.get(job.operation)
        if not handler:
            self.job_store.update_status(
                job_id, JobStatus.FAILED, error=f"No handler for {job.operation}"
            )
            return

        self.job_store.update_status(job_id, JobStatus.RUNNING, progress=0.0)

        try:
            result = await handler(job, self._progress_callback(job_id))
            self.job_store.update_status(
                job_id, JobStatus.COMPLETED, progress=1.0, result=result
            )
        except Exception as e:
            self.job_store.update_status(
                job_id, JobStatus.FAILED, error=str(e)
            )

    def _progress_callback(self, job_id: str):
        """Create a progress callback for a job."""
        def callback(progress: float):
            self.job_store.update_status(job_id, JobStatus.RUNNING, progress=progress)
        return callback

    def poll_status(self, job_id: str, timeout_seconds: int = 60, interval: float = 1.0) -> AsyncJob:
        """Poll job status until completion or timeout."""
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            job = self.job_store.get(job_id)
            if not job:
                raise ValueError(f"Job not found: {job_id}")

            print(f"[Poll] Job {job_id}: {job.status.value} ({job.progress:.0%})")

            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return job

            time.sleep(interval)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout_seconds}s")


# =============================================================================
# Async Job Tool
# =============================================================================

class AsyncJobInput(BaseModel):
    """Input for async job tool."""

    operation: str = Field(..., description="Operation to perform")
    webhook_url: Optional[str] = Field(default=None, description="Webhook for notifications")
    wait_for_completion: bool = Field(default=False, description="Wait for job to complete")


class AsyncJobTool(BaseTool):
    """
    Tool for submitting and tracking async jobs.
    """

    name: str = "Async Job Manager"
    description: str = """Submit and track asynchronous jobs.
    Supports webhook notifications and polling."""
    args_schema: Type[BaseModel] = AsyncJobInput

    job_store: JobStore = None
    executor: AsyncJobExecutor = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.job_store = JobStore()
        self.executor = AsyncJobExecutor(self.job_store)

        # Register demo handlers
        self.executor.register_handler("process_data", self._process_data_handler)
        self.executor.register_handler("generate_report", self._report_handler)

    async def _process_data_handler(self, job: AsyncJob, progress_cb: callable) -> dict:
        """Sample handler for data processing."""
        for i in range(5):
            await asyncio.sleep(0.1)  # Simulate work
            progress_cb((i + 1) / 5)
        return {"processed_records": 100, "status": "success"}

    async def _report_handler(self, job: AsyncJob, progress_cb: callable) -> dict:
        """Sample handler for report generation."""
        for i in range(3):
            await asyncio.sleep(0.1)
            progress_cb((i + 1) / 3)
        return {"report_url": "/reports/123.pdf"}

    def _run(
        self,
        operation: str,
        webhook_url: Optional[str] = None,
        wait_for_completion: bool = False,
    ) -> str:
        """Submit or check async job."""
        print(f"  [AsyncJob] Submitting: {operation}")

        # Create job
        job = self.job_store.create(operation, webhook_url)

        if wait_for_completion:
            # Execute synchronously for demo
            asyncio.run(self.executor.execute(job.job_id))
            job = self.job_store.get(job.job_id)
            return f"Job completed: {job.result}"

        return f"Job submitted: {job.job_id} (poll for status)"


# =============================================================================
# Schema Validation (CX-04)
# =============================================================================

class SchemaVersion(BaseModel):
    """Schema version metadata."""

    major: int = 1
    minor: int = 0
    patch: int = 0

    @property
    def string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


# Example domain models with strict validation

class UserInput(BaseModel):
    """Validated user input schema."""

    username: str = Field(..., min_length=3, max_length=50, pattern=r'^[a-zA-Z0-9_]+$')
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: Optional[int] = Field(default=None, ge=0, le=150)
    role: Literal["admin", "user", "guest"] = "user"

    @field_validator('username')
    @classmethod
    def username_not_reserved(cls, v):
        reserved = ['admin', 'root', 'system']
        if v.lower() in reserved:
            raise ValueError(f"Username '{v}' is reserved")
        return v


class OrderInput(BaseModel):
    """Validated order input schema."""

    product_id: str = Field(..., min_length=1)
    quantity: int = Field(..., gt=0, le=1000)
    unit_price: float = Field(..., gt=0)
    discount_percent: float = Field(default=0, ge=0, le=100)

    @property
    def total_price(self) -> float:
        return self.quantity * self.unit_price * (1 - self.discount_percent / 100)


class APIResponse(BaseModel):
    """Validated API response schema."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"

    @model_validator(mode='after')
    def check_success_has_data(self):
        if self.success and self.data is None:
            raise ValueError("Successful response must have data")
        if not self.success and self.error is None:
            raise ValueError("Failed response must have error message")
        return self


class SchemaValidator:
    """
    Schema validator for API contracts.

    Ensures type safety and validates against defined schemas.
    """

    def __init__(self):
        self.schemas: dict[str, Type[BaseModel]] = {}
        self.version = SchemaVersion()

    def register_schema(self, name: str, schema: Type[BaseModel]):
        """Register a schema by name."""
        self.schemas[name] = schema
        print(f"[Schema] Registered: {name}")

    def validate(self, schema_name: str, data: dict) -> tuple[bool, Any, Optional[str]]:
        """
        Validate data against a schema.

        Returns: (is_valid, validated_data_or_none, error_message_or_none)
        """
        if schema_name not in self.schemas:
            return False, None, f"Unknown schema: {schema_name}"

        schema = self.schemas[schema_name]

        try:
            validated = schema(**data)
            return True, validated, None
        except Exception as e:
            return False, None, str(e)

    def get_schema_json(self, schema_name: str) -> Optional[dict]:
        """Get JSON schema for documentation."""
        if schema_name not in self.schemas:
            return None
        return self.schemas[schema_name].model_json_schema()


# =============================================================================
# Schema-Validated Tool
# =============================================================================

class ValidatedInput(BaseModel):
    """Base input with schema validation."""

    schema_name: str = Field(..., description="Schema to validate against")
    data: dict = Field(..., description="Data to validate")


class SchemaValidatedTool(BaseTool):
    """
    Tool that validates input/output against schemas.
    """

    name: str = "Schema Validator"
    description: str = """Validate data against registered schemas.
    Ensures type safety and contract compliance."""
    args_schema: Type[BaseModel] = ValidatedInput

    validator: SchemaValidator = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validator = SchemaValidator()

        # Register default schemas
        self.validator.register_schema("user", UserInput)
        self.validator.register_schema("order", OrderInput)
        self.validator.register_schema("response", APIResponse)

    def _run(self, schema_name: str, data: dict) -> str:
        """Validate data against schema."""
        print(f"  [Validator] Validating against schema: {schema_name}")

        is_valid, validated, error = self.validator.validate(schema_name, data)

        if is_valid:
            return f"[VALID] Data conforms to {schema_name} schema: {validated}"
        else:
            return f"[INVALID] Validation failed: {error}"


# =============================================================================
# Contract Versioning
# =============================================================================

class ContractVersion(BaseModel):
    """API contract version with compatibility info."""

    version: str
    deprecated: bool = False
    sunset_date: Optional[datetime] = None
    breaking_changes: list[str] = []
    migration_guide: Optional[str] = None


class ContractRegistry:
    """
    Registry for API contract versions.

    Supports versioning and backward compatibility checks.
    """

    def __init__(self):
        self.contracts: dict[str, dict[str, ContractVersion]] = {}

    def register(
        self,
        contract_name: str,
        version: str,
        deprecated: bool = False,
        sunset_date: Optional[datetime] = None,
        breaking_changes: Optional[list[str]] = None,
    ):
        """Register a contract version."""
        if contract_name not in self.contracts:
            self.contracts[contract_name] = {}

        self.contracts[contract_name][version] = ContractVersion(
            version=version,
            deprecated=deprecated,
            sunset_date=sunset_date,
            breaking_changes=breaking_changes or [],
        )

        status = " (deprecated)" if deprecated else ""
        print(f"[Contract] Registered: {contract_name} v{version}{status}")

    def get_latest(self, contract_name: str) -> Optional[ContractVersion]:
        """Get the latest non-deprecated version."""
        if contract_name not in self.contracts:
            return None

        versions = self.contracts[contract_name]
        non_deprecated = [v for v in versions.values() if not v.deprecated]

        if not non_deprecated:
            return None

        return max(non_deprecated, key=lambda v: v.version)

    def check_compatibility(
        self,
        contract_name: str,
        client_version: str,
        server_version: str,
    ) -> dict:
        """Check compatibility between client and server versions."""
        result = {
            "compatible": True,
            "warnings": [],
            "breaking_changes": [],
        }

        if contract_name not in self.contracts:
            result["compatible"] = False
            result["warnings"].append(f"Unknown contract: {contract_name}")
            return result

        versions = self.contracts[contract_name]

        if client_version not in versions:
            result["warnings"].append(f"Unknown client version: {client_version}")

        if server_version not in versions:
            result["warnings"].append(f"Unknown server version: {server_version}")

        server_contract = versions.get(server_version)
        if server_contract and server_contract.deprecated:
            result["warnings"].append(
                f"Server version {server_version} is deprecated"
            )
            if server_contract.sunset_date:
                result["warnings"].append(
                    f"Sunset date: {server_contract.sunset_date}"
                )

        # Check for breaking changes between versions
        if server_contract and server_contract.breaking_changes:
            result["breaking_changes"] = server_contract.breaking_changes
            result["compatible"] = client_version == server_version

        return result


# =============================================================================
# Demonstrations
# =============================================================================

def demo_async_job():
    """Demonstrate async job pattern."""
    print("=" * 60)
    print("Demo 1: Async Job Pattern (CX-03)")
    print("=" * 60)

    store = JobStore()
    executor = AsyncJobExecutor(store)

    # Register handler
    async def demo_handler(job, progress_cb):
        for i in range(5):
            await asyncio.sleep(0.1)
            progress_cb((i + 1) / 5)
        return {"result": "Demo completed"}

    executor.register_handler("demo_operation", demo_handler)

    # Create and execute job
    job = store.create("demo_operation", webhook_url="https://webhook.example.com")
    print(f"\nCreated job: {job.job_id}")

    # Execute
    asyncio.run(executor.execute(job.job_id))

    # Check result
    completed_job = store.get(job.job_id)
    print(f"Status: {completed_job.status.value}")
    print(f"Result: {completed_job.result}")


def demo_polling():
    """Demonstrate job polling."""
    print("\n" + "=" * 60)
    print("Demo 2: Job Status Polling")
    print("=" * 60)

    store = JobStore()
    executor = AsyncJobExecutor(store)

    async def slow_handler(job, progress_cb):
        for i in range(3):
            await asyncio.sleep(0.2)
            progress_cb((i + 1) / 3)
        return {"data": "processed"}

    executor.register_handler("slow_op", slow_handler)

    job = store.create("slow_op")

    # Start execution in background
    async def run_and_poll():
        task = asyncio.create_task(executor.execute(job.job_id))

        # Poll while executing
        for _ in range(10):
            current = store.get(job.job_id)
            print(f"Poll: {current.status.value} ({current.progress:.0%})")
            if current.status == JobStatus.COMPLETED:
                break
            await asyncio.sleep(0.1)

        await task

    asyncio.run(run_and_poll())


def demo_schema_validation():
    """Demonstrate schema validation."""
    print("\n" + "=" * 60)
    print("Demo 3: Schema Validation (CX-04)")
    print("=" * 60)

    validator = SchemaValidator()
    validator.register_schema("user", UserInput)
    validator.register_schema("order", OrderInput)

    # Valid user
    print("\n--- Valid User ---")
    valid, data, error = validator.validate("user", {
        "username": "john_doe",
        "email": "john@example.com",
        "age": 30,
        "role": "user",
    })
    print(f"Valid: {valid}, Data: {data}")

    # Invalid user (reserved username)
    print("\n--- Invalid User (reserved name) ---")
    valid, data, error = validator.validate("user", {
        "username": "admin",
        "email": "admin@example.com",
    })
    print(f"Valid: {valid}, Error: {error}")

    # Invalid user (bad email)
    print("\n--- Invalid User (bad email) ---")
    valid, data, error = validator.validate("user", {
        "username": "test_user",
        "email": "not-an-email",
    })
    print(f"Valid: {valid}, Error: {error}")

    # Valid order
    print("\n--- Valid Order ---")
    valid, data, error = validator.validate("order", {
        "product_id": "PROD-123",
        "quantity": 5,
        "unit_price": 29.99,
        "discount_percent": 10,
    })
    if valid:
        print(f"Valid: {valid}, Total price: ${data.total_price:.2f}")


def demo_contract_versioning():
    """Demonstrate contract versioning."""
    print("\n" + "=" * 60)
    print("Demo 4: Contract Versioning")
    print("=" * 60)

    registry = ContractRegistry()

    # Register versions
    registry.register("user_api", "1.0.0")
    registry.register("user_api", "1.1.0")
    registry.register(
        "user_api", "2.0.0",
        breaking_changes=["Changed user ID from int to UUID"]
    )
    registry.register(
        "user_api", "1.0.0",
        deprecated=True,
        sunset_date=datetime.now() + timedelta(days=90),
    )

    # Get latest
    print("\n--- Latest Version ---")
    latest = registry.get_latest("user_api")
    print(f"Latest: v{latest.version}")

    # Check compatibility
    print("\n--- Compatibility Checks ---")

    compat = registry.check_compatibility("user_api", "1.0.0", "1.1.0")
    print(f"v1.0.0 -> v1.1.0: Compatible={compat['compatible']}")

    compat = registry.check_compatibility("user_api", "1.1.0", "2.0.0")
    print(f"v1.1.0 -> v2.0.0: Compatible={compat['compatible']}")
    if compat['breaking_changes']:
        print(f"  Breaking changes: {compat['breaking_changes']}")


def demo_async_job_tool():
    """Demonstrate async job tool."""
    print("\n" + "=" * 60)
    print("Demo 5: Async Job Tool")
    print("=" * 60)

    tool = AsyncJobTool()

    # Submit job without waiting
    print("\n--- Submit Job (no wait) ---")
    result = tool._run(operation="process_data", webhook_url="https://example.com/hook")
    print(f"Result: {result}")

    # Submit job and wait
    print("\n--- Submit Job (wait for completion) ---")
    result = tool._run(operation="generate_report", wait_for_completion=True)
    print(f"Result: {result}")


def demo_schema_validated_tool():
    """Demonstrate schema-validated tool."""
    print("\n" + "=" * 60)
    print("Demo 6: Schema-Validated Tool")
    print("=" * 60)

    tool = SchemaValidatedTool()

    # Validate user
    print("\n--- Validate User ---")
    result = tool._run(
        schema_name="user",
        data={"username": "valid_user", "email": "user@test.com"}
    )
    print(f"Result: {result}")

    # Validate order
    print("\n--- Validate Order ---")
    result = tool._run(
        schema_name="order",
        data={"product_id": "P001", "quantity": 3, "unit_price": 19.99}
    )
    print(f"Result: {result}")


def main():
    print("=" * 60)
    print("Connectors Async & Schema Verification (CX-03, CX-04)")
    print("=" * 60)
    print("""
This script verifies async job patterns and schema validation.

Verification Items:
- CX-03: Async Job Pattern
  - Job creation and tracking
  - Status polling
  - Webhook notifications
  - Progress reporting

- CX-04: Schema/Contract
  - Input validation with Pydantic
  - Output schema enforcement
  - Contract versioning
  - Backward compatibility

Key Components:
- JobStore: Async job persistence
- AsyncJobExecutor: Job execution with progress
- SchemaValidator: Pydantic-based validation
- ContractRegistry: Version management

LangGraph Comparison:
- Both support Pydantic for schema validation
- Neither has built-in async job management
- Custom implementation required for both
""")

    # Run all demos
    demo_async_job()
    demo_polling()
    demo_schema_validation()
    demo_contract_versioning()
    demo_async_job_tool()
    demo_schema_validated_tool()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
