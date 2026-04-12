import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from openai import APIConnectionError

from distill.core.llm import (AsyncLLMManager, BackendState,
                              NoHealthyBackendsError,
                              _is_loopback_base_url,
                              _scaled_concurrency_limit)
from distill.core.pipeline import DistillPipeline, TaskItem
from distill.runtime.settings import (DEFAULT_URL_TEMPLATE, PipelineConfig,
                                      resolve_base_urls)


class LoopbackBaseUrlTests(unittest.TestCase):

    def test_localhost_is_treated_as_loopback(self):
        self.assertTrue(_is_loopback_base_url("http://localhost:1597/v1"))

    def test_ipv4_loopback_is_treated_as_loopback(self):
        self.assertTrue(_is_loopback_base_url("http://127.0.0.1:1597/v1"))

    def test_ipv6_loopback_is_treated_as_loopback(self):
        self.assertTrue(_is_loopback_base_url("http://[::1]:1597/v1"))

    def test_remote_host_keeps_environment_proxy_support(self):
        self.assertFalse(
            _is_loopback_base_url("https://api.example.com/v1"))


class ResolveBaseUrlsTests(unittest.TestCase):

    def test_default_template_uses_ipv4_loopback(self):
        self.assertEqual(DEFAULT_URL_TEMPLATE, "http://127.0.0.1:{port}/v1")

    def test_ports_expand_to_ipv4_loopback_urls(self):
        self.assertEqual(
            resolve_base_urls(ports_text="1597-1598"),
            [
                "http://127.0.0.1:1597/v1",
                "http://127.0.0.1:1598/v1",
            ],
        )


class ConcurrencyScalingTests(unittest.TestCase):

    def test_scaled_concurrency_tracks_active_backend_ratio(self):
        self.assertEqual(_scaled_concurrency_limit(2048, 6, 6), 2048)
        self.assertEqual(_scaled_concurrency_limit(2048, 3, 6), 1024)
        self.assertEqual(_scaled_concurrency_limit(2048, 1, 6), 342)
        self.assertEqual(_scaled_concurrency_limit(2048, 0, 6), 0)


class _FakeCompletions:

    def __init__(self, handler):
        self._handler = handler

    async def create(self, **kwargs):
        return await self._handler(**kwargs)


class _FakeChat:

    def __init__(self, handler):
        self.completions = _FakeCompletions(handler)


class _FakeClient:

    def __init__(self, handler):
        self.chat = _FakeChat(handler)


def _build_response(content: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=None,
                    tool_calls=None,
                    tool_call_id=None,
                    name=None,
                ),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "total_tokens": 8,
        },
    )


class AsyncLLMManagerTests(unittest.IsolatedAsyncioTestCase):

    async def test_generate_stops_using_backend_only_when_vllm_ls_confirms_absent(
            self):
        config = PipelineConfig(
            input_dir="in",
            output_dir="out",
            failure_log="failure.jsonl",
            model_name="test-model",
            api_key="EMPTY",
            vllm_ls_command="/mnt/ssd/yulan/bin/vllm_ls",
            base_urls=[
                "http://127.0.0.1:20001/v1",
                "http://127.0.0.1:20002/v1",
            ],
            max_concurrency=8,
        )
        manager = AsyncLLMManager(config)

        failed_calls = 0
        successful_calls = 0

        async def fail_handler(**kwargs):
            nonlocal failed_calls
            failed_calls += 1
            raise APIConnectionError(
                message="backend down",
                request=httpx.Request("POST",
                                      "http://127.0.0.1:20001/v1/chat/completions"),
            )

        async def success_handler(**kwargs):
            nonlocal successful_calls
            successful_calls += 1
            return _build_response("ok")

        manager.backends = [
            BackendState(
                base_url="http://127.0.0.1:20001/v1",
                client=_FakeClient(fail_handler),
            ),
            BackendState(
                base_url="http://127.0.0.1:20002/v1",
                client=_FakeClient(success_handler),
            ),
        ]

        async def no_sleep(attempt: int):
            return None

        async def backend_process_present(base_url: str):
            return base_url != "http://127.0.0.1:20001/v1"

        manager._retry_sleep = no_sleep
        manager._backend_process_present = backend_process_present

        response = await manager.generate("hello")

        self.assertEqual(response["messages"][1]["content"], "ok")
        self.assertEqual(failed_calls, 1)
        self.assertEqual(successful_calls, 1)
        self.assertFalse(manager.backends[0].active)
        self.assertTrue(manager.backends[1].active)
        self.assertEqual(
            _scaled_concurrency_limit(manager.max_concurrency, 1,
                                      manager.total_backends),
            4,
        )

    async def test_generate_keeps_backend_when_vllm_ls_reports_process_alive(
            self):
        config = PipelineConfig(
            input_dir="in",
            output_dir="out",
            failure_log="failure.jsonl",
            model_name="test-model",
            api_key="EMPTY",
            vllm_ls_command="/mnt/ssd/yulan/bin/vllm_ls",
            base_urls=[
                "http://127.0.0.1:20001/v1",
                "http://127.0.0.1:20002/v1",
            ],
            max_concurrency=8,
        )
        manager = AsyncLLMManager(config)

        failed_calls = 0
        successful_calls = 0

        async def flaky_handler(**kwargs):
            nonlocal failed_calls, successful_calls
            if failed_calls == 0:
                failed_calls += 1
                raise APIConnectionError(
                    message="transient connect issue",
                    request=httpx.Request(
                        "POST",
                        "http://127.0.0.1:20001/v1/chat/completions",
                    ),
                )
            successful_calls += 1
            return _build_response("recovered")

        async def other_handler(**kwargs):
            return _build_response("other")

        manager.backends = [
            BackendState(
                base_url="http://127.0.0.1:20001/v1",
                client=_FakeClient(flaky_handler),
            ),
            BackendState(
                base_url="http://127.0.0.1:20002/v1",
                client=_FakeClient(other_handler),
            ),
        ]

        async def no_sleep(attempt: int):
            return None

        async def backend_process_present(base_url: str):
            return True

        manager._retry_sleep = no_sleep
        manager._backend_process_present = backend_process_present

        response = await manager.generate("hello")

        self.assertEqual(response["messages"][1]["content"], "recovered")
        self.assertEqual(failed_calls, 1)
        self.assertEqual(successful_calls, 1)
        self.assertTrue(manager.backends[0].active)
        self.assertTrue(manager.backends[1].active)

    async def test_generate_raises_fatal_error_when_vllm_ls_confirms_all_gone(
            self):
        config = PipelineConfig(
            input_dir="in",
            output_dir="out",
            failure_log="failure.jsonl",
            model_name="test-model",
            api_key="EMPTY",
            vllm_ls_command="/mnt/ssd/yulan/bin/vllm_ls",
            base_urls=[
                "http://127.0.0.1:20001/v1",
                "http://127.0.0.1:20002/v1",
            ],
            max_concurrency=8,
        )
        manager = AsyncLLMManager(config)

        async def fail_handler(**kwargs):
            raise APIConnectionError(
                message="backend down",
                request=httpx.Request(
                    "POST",
                    "http://127.0.0.1:20001/v1/chat/completions",
                ),
            )

        manager.backends = [
            BackendState(
                base_url="http://127.0.0.1:20001/v1",
                client=_FakeClient(fail_handler),
            ),
            BackendState(
                base_url="http://127.0.0.1:20002/v1",
                client=_FakeClient(fail_handler),
            ),
        ]

        async def no_sleep(attempt: int):
            return None

        async def backend_process_present(base_url: str):
            return False

        manager._retry_sleep = no_sleep
        manager._backend_process_present = backend_process_present

        with self.assertRaises(NoHealthyBackendsError):
            await manager.generate("hello")

        self.assertFalse(manager.backends[0].active)
        self.assertFalse(manager.backends[1].active)


class PipelineFatalBackendOutageTests(unittest.IsolatedAsyncioTestCase):

    async def test_worker_stops_pipeline_without_recording_failure_when_all_backends_die(
            self):
        config = PipelineConfig(
            input_dir="in",
            output_dir="out",
            failure_log="failure.jsonl",
            model_name="test-model",
            api_key="EMPTY",
            base_urls=["http://127.0.0.1:20001/v1"],
            max_concurrency=1,
        )
        pipeline = DistillPipeline(config)
        pipeline.failure_recorder.record_failure = AsyncMock()
        pipeline.llm_manager.generate = AsyncMock(
            side_effect=NoHealthyBackendsError("all backends gone"))

        class _Pbar:

            def update(self, n):
                return None

        await pipeline.task_queue.put(
            TaskItem(
                source_file="/tmp/input.parquet",
                source_row=12,
                rollout_index=0,
                prompt="hello",
                row_data={"question": "hello"},
            ))

        with self.assertRaises(NoHealthyBackendsError):
            await pipeline.worker(0, _Pbar())

        pipeline.failure_recorder.record_failure.assert_not_awaited()
        self.assertTrue(pipeline.stop_requested)


if __name__ == "__main__":
    unittest.main()
