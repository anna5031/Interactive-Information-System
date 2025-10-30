from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Iterable, Optional

from .service import QAServiceResult, RAGQAService


def _format_sources(sources: Iterable[str]) -> str:
    preview = list(sources)
    if not preview:
        return "없음"
    if len(preview) > 3:
        return f"{len(preview)}개"
    return f"{len(preview)}개"


def _print_result(
    result: QAServiceResult,
    *,
    show_log: bool,
    show_documents: bool,
    dump_json: bool,
    log_already_streamed: bool,
) -> None:
    answer = result.answer or "(응답 생성 실패)"
    confidence = f"{result.confidence:.2f}"
    print("\n🧠 질의 처리 결과")
    print(f"   답변: {answer}")
    print(f"   처리 시간: {result.processing_time:.2f}s")
    print(f"   의도: {result.intent or 'unknown'} (신뢰도: {confidence})")
    status = "통과" if result.is_safe else f"차단({', '.join(result.blocked_by) or '?'})"
    print(f"   필터 결과: {status}")
    print(f"   참조 문서: {_format_sources(result.sources)}")

    if show_documents and result.sources:
        print("   ▶ 참조 문서 미리보기:")
        for idx, doc in enumerate(result.sources[:3], start=1):
            print(f"      [{idx}] {doc}")

    print(f"   경로 안내 필요: {'예' if result.needs_navigation else '아니오'}")
    if result.map_path:
        print(f"   경로 지도: {result.map_path}")

    if show_log and not log_already_streamed:
        if result.processing_log:
            print("\n📋 Processing Log:")
            for entry in result.processing_log:
                print(f" - {entry}")
        else:
            print("\n📋 Processing Log: 없음")

    if dump_json:
        print("\n🔎 Raw response:")
        print(json.dumps(result.raw, indent=2, ensure_ascii=False))


async def _handle_question(
    service: RAGQAService,
    question: str,
    *,
    show_log: bool,
    show_documents: bool,
    dump_json: bool,
    stream_log: bool,
) -> None:
    print(f"\n❓ 질문: {question}")
    try:
        result = await service.query(question, emit_processing_log=stream_log)
    except Exception as exc:
        print(f"❌ 질의 처리 중 예외가 발생했습니다: {exc}")
        return

    _print_result(
        result,
        show_log=show_log,
        show_documents=show_documents,
        dump_json=dump_json,
        log_already_streamed=stream_log,
    )


async def run_cli_async(
    question: Optional[str] = None,
    *,
    interactive: bool = False,
    show_log: bool = False,
    show_documents: bool = False,
    dump_json: bool = False,
    stream_log: bool = False,
    include_navigation_warmup: bool = False,
) -> None:
    """비동기 CLI 루프. 테스트 및 직접 실행용."""
    service = RAGQAService(default_emit_processing_log=False)

    if include_navigation_warmup:
        await service.warm_up(include_navigation=True)
    else:
        await service.ensure_ready()

    is_interactive = interactive or question is None
    if is_interactive:
        print("🛠 개발용 QA CLI (종료하려면 exit/quit)")

    async def _single(question_text: str) -> None:
        await _handle_question(
            service,
            question_text,
            show_log=show_log,
            show_documents=show_documents,
            dump_json=dump_json,
            stream_log=stream_log,
        )

    if is_interactive:
        while True:
            try:
                user_input = input("\n질문> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 CLI를 종료합니다.")
                break

            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered in {"exit", "quit"}:
                print("👋 CLI를 종료합니다.")
                break

            await _single(user_input)
    else:
        await _single(question)


def run_cli(
    question: Optional[str] = None,
    *,
    interactive: bool = False,
    show_log: bool = False,
    show_documents: bool = False,
    dump_json: bool = False,
    stream_log: bool = False,
    include_navigation_warmup: bool = False,
) -> None:
    """동기 어댑터."""
    asyncio.run(
        run_cli_async(
            question,
            interactive=interactive,
            show_log=show_log,
            show_documents=show_documents,
            dump_json=dump_json,
            stream_log=stream_log,
            include_navigation_warmup=include_navigation_warmup,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="기계공학과 QA RAG 파이프라인을 CLI에서 실행합니다.",
    )
    parser.add_argument("-q", "--question", help="단일 질문을 실행합니다.")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="대화형 CLI 모드로 실행합니다.",
    )
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="워크플로우 처리 로그를 요약 형태로 출력합니다.",
    )
    parser.add_argument(
        "--show-documents",
        action="store_true",
        help="Retrieval된 상위 3개 문서를 출력합니다.",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="최종 결과를 JSON 형식으로 출력합니다.",
    )
    parser.add_argument(
        "--stream-log",
        action="store_true",
        help="서비스 단계별 로그를 즉시 출력합니다.",
    )
    parser.add_argument(
        "--include-navigation-warmup",
        action="store_true",
        help="초기화 시 내비게이션 모듈까지 선행 로드합니다.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.interactive and not args.question:
        parser.error("질문을 입력하거나 --interactive 모드를 사용하세요.")

    try:
        run_cli(
            question=args.question,
            interactive=args.interactive,
            show_log=args.show_log,
            show_documents=args.show_documents,
            dump_json=args.dump_json,
            stream_log=args.stream_log,
            include_navigation_warmup=args.include_navigation_warmup,
        )
    except KeyboardInterrupt:
        print("\n👋 작업이 취소되었습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
