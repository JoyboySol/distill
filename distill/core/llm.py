import asyncio
import logging
from typing import Any, Dict, List, Optional

from openai import (APIConnectionError, APITimeoutError, AsyncOpenAI,
                    RateLimitError)
from tenacity import (RetryError, before_sleep_log, retry,
                      retry_if_exception_type, retry_if_not_exception_type,
                      stop_after_attempt, wait_exponential)

try:
    from ..runtime.settings import PipelineConfig, logger
    from ..common.utils import ensure_message_shape, usage_to_dict
except ImportError:
    from runtime.settings import PipelineConfig, logger
    from common.utils import ensure_message_shape, usage_to_dict


class GenerationResponse(Dict[str, Any]):
    pass


class AsyncLLMManager:
    def __init__(self, config: PipelineConfig):
        self.model = config.model_name
        self.timeout = config.llm_timeout
        self.max_tokens = config.llm_max_tokens
        self.base_urls = list(config.base_urls)
        if not self.base_urls:
            raise ValueError("AsyncLLMManager requires at least one base URL.")
        self.clients = [
            AsyncOpenAI(api_key=config.api_key, base_url=url, max_retries=0)
            for url in self.base_urls
        ]
        self.semaphore = asyncio.Semaphore(config.max_concurrency)
        self.pending_counts = [0] * len(self.clients)

    @retry(
        retry=(
            retry_if_exception_type((APIConnectionError, RateLimitError))
            & retry_if_not_exception_type(APITimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def generate_with_retry(self,
                                  prompt: str) -> Optional[GenerationResponse]:
        min_count = min(self.pending_counts)
        client_idx = self.pending_counts.index(min_count)
        client = self.clients[client_idx]
        client_url = self.base_urls[client_idx]
        self.pending_counts[client_idx] += 1

        try:
            async with self.semaphore:
                try:
                    response = await client.chat.completions.create(
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
                except Exception as exc:
                    logger.warning(
                        "LLM request failed on %s with %s: %s",
                        client_url,
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
            self.pending_counts[client_idx] -= 1

    async def generate(self, prompt: str) -> Optional[GenerationResponse]:
        try:
            return await self.generate_with_retry(prompt)
        except RetryError as exc:
            last_exc = exc.last_attempt.exception()
            if last_exc is not None:
                raise last_exc
            raise
