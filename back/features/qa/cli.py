from __future__ import annotations

import argparse
import asyncio
import logging

from rag_system import StreamingRAGService

from .pipeline import QAPipeline
from .services.voice_io import create_voice_service

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QA 음성 파이프라인 실행 도구")
    parser.add_argument(
        "--greeting",
        default="안녕하세요. 무엇을 도와드릴까요?",
        help="파이프라인 시작 시 재생할 인사 멘트 (빈 문자열이면 생략)",
    )
    parser.add_argument(
        "--closing-message",
        default="필요하시면 다시 불러주세요.",
        help="세션 종료 시 재생할 멘트",
    )
    parser.add_argument(
        "--followup-timeout",
        type=float,
        default=None,
        help="첫 질문 이후 후속 질문을 기다리는 시간(초). 미지정 시 SessionConfig 기본값 사용",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="TTS 인사 멘트를 건너뜁니다.",
    )
    return parser


async def run_once(args: argparse.Namespace) -> None:
    greeting_text = "" if args.silent else args.greeting
    voice_service = create_voice_service()
    rag_service = StreamingRAGService()
    pipeline = QAPipeline(
        voice_service=voice_service,
        rag_service=rag_service,
        greeting=greeting_text,
        closing_message=args.closing_message,
        followup_idle_timeout=args.followup_timeout,
    )
    await pipeline.run()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    try:
        asyncio.run(run_once(args))
    except KeyboardInterrupt:
        logger.info("사용자 입력으로 중단되었습니다.")


if __name__ == "__main__":
    main()
