from __future__ import annotations

import argparse
import sys
from pathlib import Path

from services.navigation_service import IndoorNavigationService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _format_node_description(node: dict) -> str:
    floor_label = node.get("floor_label") or node.get("floor_value") or node.get("request_id")
    base = f"[{floor_label}] {node.get('node_id')}"
    node_type = node.get("type") or node.get("category") or ""
    description = f"{base} ({node_type})" if node_type else base
    if node.get("name"):
        description = f"{description} · {node['name']}"
    elif node.get("number"):
        description = f"{description} · {node['number']}"
    return description


def _print_result(building_id: str, start: str, destination: str, result: dict) -> None:
    print(f"건물 ID: {building_id}")
    print(f"출발 노드: {start}")
    print(f"도착 노드: {destination}")
    print("")

    print("경로 노드 순서:")
    for index, node in enumerate(result.get("nodes", []), start=1):
        print(f"  {index}. {_format_node_description(node)}")
    print("")

    print("세부 엣지 목록:")
    if not result.get("edges"):
        print("  - 연결된 엣지가 없습니다.")
    else:
        for edge in result.get("edges", []):
            edge_type = "층간 이동" if edge.get("is_cross_floor") else "일반"
            source_label = edge.get("from_request_id")
            target_label = edge.get("to_request_id")
            source = f"{source_label}:{edge.get('from')}"
            target = f"{target_label}:{edge.get('to')}"
            duration = edge.get("duration_seconds")
            duration_text = f" · 시간 {duration:.1f}s" if isinstance(duration, (int, float)) else ""
            print(f"  - {edge_type}: {source} -> {target} · 가중치 {edge.get('weight', 0.0):.2f}{duration_text}")
    print("")

    weight_summary = result.get("weight_summary") or {}
    normal_total = weight_summary.get("normal_total", 0.0)
    cross_total = weight_summary.get("cross_floor_total", 0.0)
    print(f"일반 엣지 가중치 합계: {normal_total:.2f}")
    if result.get("cross_floor_segments"):
        print("층간 이동 엣지:")
        for segment in result["cross_floor_segments"]:
            source_label = segment.get("from_request_id")
            target_label = segment.get("to_request_id")
            source = f"{source_label}:{segment.get('from')}"
            target = f"{target_label}:{segment.get('to')}"
            duration = segment.get("duration_seconds")
            duration_text = f" · 시간 {duration:.1f}s" if isinstance(duration, (int, float)) else ""
            print(f"  - {source} -> {target} · 가중치 {segment.get('weight', 0.0):.2f}{duration_text}")
    else:
        print("층간 이동 엣지: 없음")
    print(f"전체 가중치 합계: {weight_summary.get('overall', normal_total + cross_total):.2f}")
    duration_summary = result.get("duration_summary") or {}
    horizontal_time = duration_summary.get("horizontal_total", 0.0)
    vertical_time = duration_summary.get("vertical_total", 0.0)
    overall_time = duration_summary.get("overall", horizontal_time + vertical_time)
    print(f"수평 이동 시간 합계: {horizontal_time:.1f}s")
    print(f"층간 이동 시간 합계: {vertical_time:.1f}s")
    print(f"총 예상 소요 시간: {overall_time:.1f}s")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="실내 지도 경로 탐색 CLI")
    parser.add_argument("--building", "-b", required=True, help="건물 ID (data/<id> 폴더)")
    parser.add_argument("--start", "-s", required=True, help="출발 노드 (floorId:nodeId 형식 권장)")
    parser.add_argument("--end", "-e", required=True, help="도착 노드 (floorId:nodeId 형식 권장)")
    parser.add_argument("--data-root", default=str(DATA_DIR), help="데이터 루트 경로 (기본값: 프로젝트 data 디렉터리)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    service = IndoorNavigationService(Path(args.data_root))
    try:
        result = service.find_shortest_path(args.building, args.start, args.end)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"경로 탐색에 실패했습니다: {exc}", file=sys.stderr)
        return 1
    _print_result(args.building, args.start, args.end, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
