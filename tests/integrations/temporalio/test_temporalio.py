# test_temporalio.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment

import asyncio
from datetime import timedelta
from typing import Protocol
from uuid import uuid4

from dishka import Provider, Scope, make_async_container, FromDishka, AsyncContainer
from dishka.integrations.temporalio import DishkaWorkerInterceptor
from temporalio import activity, workflow
from temporalio.worker import Worker


# --------------- Application Components ---------------
class MyService(Protocol):
    async def perform_business_logic(self, user_id: str) -> str: ...


class MyServiceImpl(MyService):
    async def perform_business_logic(self, user_id: str) -> str:
        return f"Business logic executed for user {user_id}"


# --------------- Dependency Injection Setup ---------------
def create_container() -> AsyncContainer:
    service_provider = Provider(scope=Scope.REQUEST)
    service_provider.provide(MyServiceImpl, provides=MyService)
    return make_async_container(service_provider)


# --------------- Temporal Activities ---------------
@activity.defn
async def do_something(
    user_id: str,
    service: FromDishka[MyService],  # Auto-injected
) -> str:
    return await service.perform_business_logic(user_id)


# --------------- Workflows ---------------
@workflow.defn
class MyWorkflow:
    @workflow.run
    async def run(self, user_id: str) -> str:
        return await workflow.execute_activity(
            do_something,
            args=[user_id],
            schedule_to_close_timeout=timedelta(seconds=30),
        )

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_env():
    env = await WorkflowEnvironment.start_time_skipping()
    yield env
    await env.shutdown()


@pytest.fixture
def mock_service():
    mock = AsyncMock(spec=MyService)
    mock.perform_business_logic.return_value = "mock result"
    return mock


@pytest.fixture
def test_container(mock_service):
    # Override the real container with test dependencies
    provider = Provider(Scope.REQUEST)
    provider.provide(lambda: mock_service, provides=MyService)
    return make_async_container(provider)


@pytest.fixture
async def worker(test_env, test_container):
    async with Worker(
            test_env.client,
            task_queue="test-task-queue",
            workflows=[MyWorkflow],
            activities=[do_something],
            interceptors=[DishkaWorkerInterceptor(test_container)],
    ):
        yield


async def test_activity_execution(test_env, worker, mock_service):
    user_id = "test-user-123"
    result = await test_env.client.execute_activity(
        do_something,
        user_id,
        task_queue="test-task-queue",
        schedule_to_close_timeout=timedelta(seconds=10),
    )

    mock_service.perform_business_logic.assert_awaited_once_with(user_id)
    assert result == "mock result"


async def test_workflow_execution(test_env, worker, mock_service):
    user_id = "workflow-user-456"
    result = await test_env.client.execute_workflow(
        MyWorkflow.run,
        user_id,
        id=f"test-workflow-{uuid4()}",
        task_queue="test-task-queue",
    )

    mock_service.perform_business_logic.assert_awaited_once_with(user_id)
    assert result == "mock result"


async def test_missing_dependency(test_env, test_container):
    # Test with broken container configuration
    broken_provider = Provider(Scope.REQUEST)
    broken_container = make_async_container(broken_provider)

    async with Worker(
            test_env.client,
            task_queue="broken-task-queue",
            workflows=[MyWorkflow],
            activities=[do_something],
            interceptors=[DishkaWorkerInterceptor(broken_container)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await test_env.client.execute_workflow(
                MyWorkflow.run,
                "should-fail",
                id=f"fail-workflow-{uuid4()}",
                task_queue="broken-task-queue",
            )

        assert "No provider for" in str(exc_info.value.__cause__)


def test_interceptor_wrapping():
    interceptor = DishkaWorkerInterceptor(MagicMock())
    next_interceptor = MagicMock()
    activity_interceptor = interceptor.intercept_activity(next_interceptor)

    assert isinstance(activity_interceptor, DishkaActivityInboundInterceptor)
    assert activity_interceptor.next == next_interceptor


@activity.defn
async def typed_activity(
        service: FromDishka[MyService],
        normal_param: str,
) -> str:
    return await service.perform_business_logic(normal_param)


async def test_typed_parameters(test_env, test_container, mock_service):
    async with Worker(
            test_env.client,
            task_queue="typed-task-queue",
            activities=[typed_activity],
            interceptors=[DishkaWorkerInterceptor(test_container)],
    ):
        result = await test_env.client.execute_activity(
            typed_activity,
            "typed-param",
            task_queue="typed-task-queue",
            schedule_to_close_timeout=timedelta(seconds=10),
        )

        mock_service.perform_business_logic.assert_awaited_once_with("typed-param")
        assert result == "mock result"


async def test_scoped_dependencies(test_env, mock_service):
    # Verify container scoping works correctly
    call_count = 0

    async def scoped_service_provider():
        nonlocal call_count
        call_count += 1
        return mock_service

    provider = Provider(Scope.REQUEST)
    provider.provide(scoped_service_provider, provides=MyService)
    container = make_async_container(provider)

    async with Worker(
            test_env.client,
            task_queue="scoped-task-queue",
            activities=[do_something],
            interceptors=[DishkaWorkerInterceptor(container)],
    ):
        # Execute two activities - should create two scoped containers
        await test_env.client.execute_activity(
            do_something,
            "user1",
            task_queue="scoped-task-queue",
        )
        await test_env.client.execute_activity(
            do_something,
            "user2",
            task_queue="scoped-task-queue",
        )

        assert call_count == 2
