import argparse
import asyncio
import logging
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features.exploration.pipeline import ExplorationPipeline
from features.exploration.config import DetectionConfig


SOURCE = 0
MAX_FRAMES = None  # 정수로 제한을 두고 싶다면 설정


async def main(camera_source, max_frames, detection_config: DetectionConfig):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    pipeline = ExplorationPipeline(
        camera_source=camera_source,
        show_overlay=True,
        max_frames=max_frames,
        detection_config=detection_config,
    )
    try:
        await pipeline.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual exploration pipeline test")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override camera index or video file path (default: SOURCE constant)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Override max frame limit (default: MAX_FRAMES constant)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold override",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU threshold override",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=None,
        help="Keypoint confidence threshold override",
    )
    args = parser.parse_args()

    source = args.source if args.source is not None else SOURCE
    if source is None:
        camera_source = 0
    else:
        if isinstance(source, str) and source.isdigit():
            camera_source = int(source)
        else:
            camera_source = source

    max_frames = args.max_frames if args.max_frames is not None else MAX_FRAMES

    default_cfg = DetectionConfig()
    detection_config = DetectionConfig(
        confidence_threshold=(
            args.conf if args.conf is not None else default_cfg.confidence_threshold
        ),
        iou_threshold=args.iou if args.iou is not None else default_cfg.iou_threshold,
        keypoint_confidence_threshold=(
            args.kp
            if args.kp is not None
            else default_cfg.keypoint_confidence_threshold
        ),
    )

    asyncio.run(
        main(
            camera_source=camera_source,
            max_frames=max_frames,
            detection_config=detection_config,
        )
    )
