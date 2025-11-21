from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from processing.wall_centerline_pipeline import WallCenterlinePipeline, WallPipelineDebugConfig


def main(args: argparse.Namespace) -> None:
    """이미지와 선택적 힌트 박스를 받아 파이프라인을 실행하고 디버그 이미지를 저장한다."""
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image)

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or image_path.stem

    pipeline = WallCenterlinePipeline()
    debug_config = WallPipelineDebugConfig(output_dir=output_dir, prefix=prefix)
    result = pipeline.run(image_array, debug_config=debug_config)

    print(f"[wall-pipeline] raw={len(result.raw_lines)} final={len(result.final_lines)}")
    if result.debug_assets:
        print("[wall-pipeline] 디버그 이미지:")
        for path in result.debug_assets:
            print(" -", path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """명령행 인자를 파싱해 이미지/출력 경로 등을 반환한다."""
    parser = argparse.ArgumentParser(description="LSD 기반 벽 중심선 파이프라인 디버깅 실행기")
    parser.add_argument("--image", required=True, help="분석할 도면 이미지 경로")
    parser.add_argument(
        "--output",
        default="data/wall_pipeline_debug",
        help="디버그 이미지를 저장할 디렉터리 (기본값: data/wall_pipeline_debug)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="디버그 파일명 접두사 (기본: 이미지 파일명)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
