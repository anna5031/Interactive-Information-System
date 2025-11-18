from __future__ import annotations

import asyncio
from typing import Callable


async def request_rephrase(retry_prompt: str = "질문을 잘 인식하지 못했어요. 다시 말씀해 주세요.") -> str:
    print(retry_prompt)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: input("[마이크] "))


async def with_retry(handler: Callable[[str], "asyncio.Future[str]"], prompt: str) -> str:
    question = prompt
    while True:
        result = await handler(question)
        if result:
            return result
        question = await request_rephrase()
