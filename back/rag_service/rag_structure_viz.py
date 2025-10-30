from __future__ import annotations

from pathlib import Path

from .info_system import UniversityMEInfoSystem


def _build_workflow_without_dependencies():
    """데이터/임베딩 초기화 없이 워크플로우 그래프만 구성."""
    dummy = object.__new__(UniversityMEInfoSystem)
    dummy.use_guardrail = False
    dummy.enable_intent_rewrite = False
    return UniversityMEInfoSystem._create_workflow(dummy)


def export_langgraph_png(output_path: Path | None = None) -> Path:
    """LangGraph가 제공하는 기본 draw 기능으로 워크플로우 이미지를 생성."""
    if output_path is None:
        output_path = Path(__file__).parent / "rag_workflow.png"
    else:
        output_path = Path(output_path)

    workflow = _build_workflow_without_dependencies()
    graph = workflow.get_graph()

    try:
        graph.draw_png(str(output_path))
    except Exception:
        # draw_png는 graphviz가 필요. 없을 경우 mermaid 기반으로 대체
        alt_path = output_path.with_suffix(".mermaid.png")
        graph.draw_mermaid_png(output_file_path=str(alt_path))
        output_path = alt_path

    return output_path


if __name__ == "__main__":
    path = export_langgraph_png()
    print(f"✅ LangGraph 워크플로우 이미지 생성 완료: {path}")
