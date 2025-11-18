from __future__ import annotations

import argparse
import asyncio

from .service import StreamingRAGService, build_index_from_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Back RAG 시스템 CLI")
    parser.add_argument("--build-index", action="store_true", help="텍스트 문서로부터 인덱스를 재구성")
    parser.add_argument("--interactive", action="store_true", help="콘솔 QA 세션 실행")
    parser.add_argument("--question", help="단일 질문 수행")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.build_index:
        build_index_from_source()
        print("✅ 인덱스를 재구축했습니다.")
    service: StreamingRAGService | None = None
    if args.interactive:
        service = service or StreamingRAGService()
        asyncio.run(service.run_interactive())
        return
    if args.question:
        service = service or StreamingRAGService()
        result = asyncio.run(service.answer(args.question))
        print(result.answer)
        StreamingRAGService._print_similarity(result)
        if result.navigation.get("success"):
            print("경로 안내:", result.navigation["message"])
        elif result.navigation_request.get("destination"):
            dest = result.navigation_request.get("destination")
            origin = "예시 시작점"
            print(f"경로 안내 대기: 출발 {origin or '알 수 없음'} → 도착 {dest}")
    elif not args.build_index:
        parser.error("--question, --interactive 중 하나를 선택하거나 --build-index를 사용하세요.")


if __name__ == "__main__":
    main()
