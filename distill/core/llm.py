import asyncio
import ipaddress
import math
import re
import shlex
import subprocess
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import APIConnectionError, AsyncOpenAI, RateLimitError

try:
    from ..runtime.settings import PipelineConfig, logger
    from ..common.utils import ensure_message_shape, usage_to_dict
except ImportError:
    from runtime.settings import PipelineConfig, logger
    from common.utils import ensure_message_shape, usage_to_dict


class GenerationResponse(Dict[str, Any]):
    pass


class NoHealthyBackendsError(RuntimeError):
    pass


@dataclass
class BackendState:
    base_url: str
    client: AsyncOpenAI
    active: bool = True
    pending_count: int = 0


def _is_loopback_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    hostname = parsed.hostname
    if hostname is None:
        return False
    if hostname in {"localhost", "0.0.0.0"}:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _scaled_concurrency_limit(base_limit: int, active_backends: int,
                              total_backends: int) -> int:
    if base_limit <= 0:
        raise ValueError("base_limit must be positive")
    if total_backends <= 0:
        raise ValueError("total_backends must be positive")
    if active_backends <= 0:
        return 0
    return min(base_limit,
               max(1, math.ceil(base_limit * active_backends / total_backends)))


class AsyncLLMManager:
    def __init__(self, config: PipelineConfig):
        self.model = config.model_name
        self.timeout = config.llm_timeout
        self.max_tokens = config.llm_max_tokens
        self.base_urls = list(config.base_urls)
        if not self.base_urls:
            raise ValueError("AsyncLLMManager requires at least one base URL.")
        self.backends = [
            BackendState(
                base_url=url,
                client=AsyncOpenAI(
                    api_key=config.api_key,
                    base_url=url,
                    max_retries=0,
                    http_client=httpx.AsyncClient(
                        trust_env=not _is_loopback_base_url(url)),
                ),
            ) for url in self.base_urls
        ]
        self.max_concurrency = config.max_concurrency
        self.total_backends = len(self.backends)
        self.vllm_ls_command = config.vllm_ls_command
        self._inflight_requests = 0
        self._state_cond = asyncio.Condition()

    async def _acquire_capacity(self):
        async with self._state_cond:
            while True:
                active_count = sum(1 for backend in self.backends if backend.active)
                if active_count == 0:
                    raise NoHealthyBackendsError(
                        "No healthy LLM backends remain available.")
                limit = _scaled_concurrency_limit(self.max_concurrency,
                                                  active_count,
                                                  self.total_backends)
                if self._inflight_requests < limit:
                    self._inflight_requests += 1
                    return
                await self._state_cond.wait()

    async def _release_capacity(self):
        async with self._state_cond:
            if self._inflight_requests > 0:
                self._inflight_requests -= 1
            self._state_cond.notify_all()

    async def _acquire_backend(self) -> Tuple[int, BackendState]:
        async with self._state_cond:
            active_indices = [
                idx for idx, backend in enumerate(self.backends) if backend.active
            ]
            if not active_indices:
                raise NoHealthyBackendsError(
                    "No healthy LLM backends remain available.")
            backend_idx = min(active_indices,
                              key=lambda idx: self.backends[idx].pending_count)
            backend = self.backends[backend_idx]
            backend.pending_count += 1
            return backend_idx, backend

    async def _release_backend(self, backend_idx: int):
        async with self._state_cond:
            backend = self.backends[backend_idx]
            if backend.pending_count > 0:
                backend.pending_count -= 1
            self._state_cond.notify_all()

    async def _mark_backend_unhealthy(self, backend_idx: int,
                                      exc: Exception) -> int:
        async with self._state_cond:
            backend = self.backends[backend_idx]
            if not backend.active:
                return sum(1 for item in self.backends if item.active)
            backend.active = False
            active_count = sum(1 for item in self.backends if item.active)
            new_limit = _scaled_concurrency_limit(self.max_concurrency,
                                                  active_count,
                                                  self.total_backends)
            logger.error(
                "Marked LLM backend unhealthy: %s (%s: %s). Active backends: %s/%s. Effective concurrency: %s/%s",
                backend.base_url,
                type(exc).__name__,
                exc,
                active_count,
                self.total_backends,
                new_limit,
                self.max_concurrency,
            )
            self._state_cond.notify_all()
            return active_count

    def _port_for_base_url(self, base_url: str) -> Optional[int]:
        parsed = urlparse(base_url)
        try:
            return parsed.port
        except ValueError:
            return None

    def _run_vllm_ls(self) -> Optional[str]:
        if not self.vllm_ls_command:
            return None
        try:
            result = subprocess.run(
                shlex.split(self.vllm_ls_command),
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError:
            logger.warning("vLLM health check command not found: %s",
                           self.vllm_ls_command)
            return None
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "vLLM health check command failed with exit code %s: %s",
                exc.returncode,
                self.vllm_ls_command,
            )
            if exc.stdout:
                logger.warning("vLLM health check stdout: %s",
                               exc.stdout.strip())
            if exc.stderr:
                logger.warning("vLLM health check stderr: %s",
                               exc.stderr.strip())
            return None
        return result.stdout

    def _ports_seen_in_vllm_ls_output(self, output: str) -> List[int]:
        ports = set()
        for match in re.finditer(r"(?:^|\s)--port\s+(\d+)(?:\s|$)", output):
            try:
                ports.add(int(match.group(1)))
            except (TypeError, ValueError):
                continue
        return sorted(ports)

    def _port_seen_in_vllm_ls_output(self, output: str, port: int) -> bool:
        return port in self._ports_seen_in_vllm_ls_output(output)

    async def _backend_process_present(self, base_url: str) -> Optional[bool]:
        if not self.vllm_ls_command:
            logger.debug(
                "Skipping backend process verification for %s because vllm_ls command is not configured",
                base_url,
            )
            return None
        if not _is_loopback_base_url(base_url):
            logger.debug(
                "Skipping backend process verification for non-loopback backend %s",
                base_url,
            )
            return None

        port = self._port_for_base_url(base_url)
        if port is None:
            logger.warning("Could not extract port from base URL: %s", base_url)
            return None

        output = await asyncio.to_thread(self._run_vllm_ls)
        if output is None:
            logger.warning(
                "vLLM health check was inconclusive for %s because command output was unavailable",
                base_url,
            )
            return None
        ports_seen = self._ports_seen_in_vllm_ls_output(output)
        if not ports_seen:
            logger.warning(
                "vLLM health check was inconclusive for %s because no '--port' entries were parsed from command output",
                base_url,
            )
            return None
        if port in ports_seen:
            logger.info(
                "Verified backend %s as alive via vllm_ls (matched port %s)",
                base_url,
                port,
            )
            return True
        logger.warning(
            "Backend %s was not found in vllm_ls output. Expected port %s; visible ports: %s",
            base_url,
            port,
            ports_seen,
        )
        return False

    async def _retry_sleep(self, attempt: int):
        delay = min(max(2**attempt, 2), 10)
        logger.warning("Retrying LLM request in %ss", delay)
        await asyncio.sleep(delay)

    async def generate_with_retry(self,
                                  prompt: str) -> Optional[GenerationResponse]:
        last_exc: Optional[Exception] = None
        max_attempts = 5

        for attempt in range(1, max_attempts + 1):
            backend_idx: Optional[int] = None

            await self._acquire_capacity()
            try:
                backend_idx, backend = await self._acquire_backend()
                try:
                    response = await backend.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                        top_p=0.95,
                        extra_body={
                            "chat_template_kwargs": {
                                "enable_thinking": False
                            }
                        },
                    )
                except APIConnectionError as exc:
                    last_exc = exc
                    logger.warning(
                        "LLM request failed on %s with %s: %s",
                        backend.base_url,
                        type(exc).__name__,
                        exc,
                    )
                    backend_present = await self._backend_process_present(
                        backend.base_url)
                    if backend_present is False:
                        active_count = await self._mark_backend_unhealthy(
                            backend_idx, exc)
                        if active_count == 0:
                            raise NoHealthyBackendsError(
                                "All configured vLLM backends disappeared from vllm_ls output."
                            ) from exc
                    elif backend_present is True:
                        logger.warning(
                            "Keeping backend active after APIConnectionError because vllm_ls still reports it alive: %s",
                            backend.base_url,
                        )
                    else:
                        if self.vllm_ls_command:
                            logger.warning(
                                "Could not verify backend process state for %s; keeping it active and retrying",
                                backend.base_url,
                            )
                        else:
                            logger.warning(
                                "Could not verify backend process state for %s because no vllm_ls_command is configured; keeping it active and retrying",
                                backend.base_url,
                            )
                    if attempt < max_attempts:
                        await self._retry_sleep(attempt)
                        continue
                    raise
                except RateLimitError as exc:
                    last_exc = exc
                    logger.warning(
                        "LLM request rate-limited on %s with %s: %s",
                        backend.base_url,
                        type(exc).__name__,
                        exc,
                    )
                    if attempt < max_attempts:
                        await self._retry_sleep(attempt)
                        continue
                    raise
                except Exception as exc:
                    logger.warning(
                        "LLM request failed on %s with %s: %s",
                        backend.base_url,
                        type(exc).__name__,
                        exc,
                    )
                    raise

                choice = response.choices[0]
                if choice.message.content is None:
                    logger.warning("Got None content! Raw response %s", choice)
                    return None

                assistant_msg = ensure_message_shape({
                    "role": "assistant",
                    "content": choice.message.content,
                    "reasoning_content": getattr(choice.message,
                                                 "reasoning_content", None),
                    "tool_calls": getattr(choice.message, "tool_calls", None),
                    "tool_call_id": getattr(choice.message, "tool_call_id", None),
                    "name": getattr(choice.message, "name", None),
                })

                user_msg = ensure_message_shape({
                    "role": "user",
                    "content": prompt,
                    "reasoning_content": None,
                    "tool_calls": None,
                    "tool_call_id": None,
                    "name": None,
                })
                return {
                    "messages": [user_msg, assistant_msg],
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "usage": usage_to_dict(getattr(response, "usage", None)),
                }
            finally:
                if backend_idx is not None:
                    await self._release_backend(backend_idx)
                await self._release_capacity()

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM generation failed without a captured exception")

    async def generate(self, prompt: str) -> Optional[GenerationResponse]:
        return await self.generate_with_retry(prompt)
