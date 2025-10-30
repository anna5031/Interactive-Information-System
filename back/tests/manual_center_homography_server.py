from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import websockets

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.events import MotorStateEvent
from features.homography import HomographyCalculator, load_calibration_bundle, PixelToWorldMapper


def _build_center_homography() -> list[list[float]]:
    config = load_config()
    bundle = load_calibration_bundle(
        config.homography.files.camera_calibration_file,
        config.homography.files.camera_extrinsics_file,
    )
    mapper = PixelToWorldMapper(bundle)

    cx = float(bundle.intrinsics.matrix[0, 2])
    cy = float(bundle.intrinsics.matrix[1, 2])
    foot_world = mapper.pixel_to_world(
        (cx, cy),
        plane_z=config.homography.floor_z_mm,
    )
    if foot_world is None:
        raise RuntimeError("센터 픽셀을 월드 좌표로 변환할 수 없습니다. 캘리브레이션을 확인하세요.")

    foot_world_arr = np.array(foot_world, dtype=float)
    projector = np.array(config.homography.projector_position_mm, dtype=float)

    vector = foot_world_arr - projector
    vector[2] = 0.0
    norm = np.linalg.norm(vector[:2])
    if norm < 1e-6:
        direction = np.array([0.0, 1.0, 0.0])
    else:
        direction = vector / norm

    target = foot_world_arr.copy()
    target[:2] = target[:2] + direction[:2] * config.motor.geometry.projection_ahead_mm
    target[2] = config.homography.floor_z_mm

    motor_state = MotorStateEvent(
        pan=0.0,
        tilt=0.0,
        has_target=True,
        timestamp=time.time(),
        head_position=None,
        foot_position=None,
        direction_label=None,
        target_pixel=(cx, cy),
        head_pixel=None,
        foot_pixel=(cx, cy),
        world_target=tuple(float(v) for v in target),
        foot_world=tuple(float(v) for v in foot_world_arr),
        distance_to_projector=float(np.linalg.norm(foot_world_arr - projector)),
        approach_velocity_mm_s=None,
        is_approaching=None,
    )

    calculator = HomographyCalculator(
        calibration=bundle,
        config=config.homography,
    )
    event = calculator.build(motor_state)
    return event.matrix


async def _serve(interval: float = 1.0) -> None:
    config = load_config()
    matrix = _build_center_homography()
    try:
        matrix_inv = np.linalg.inv(np.asarray(matrix))
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("계산된 호모그래피가 역행렬을 가지지 않습니다.") from exc
    matrix_inv_list = matrix_inv.tolist()

    async def handler(websocket: websockets.WebSocketServerProtocol) -> None:
        print("클라이언트 연결:", websocket.remote_address)
        await websocket.send(
            json.dumps(
                {
                    "type": "sync",
                    "state": {"phase": "testing"},
                    "timestamp": time.time(),
                }
            )
        )
        command = {
            "type": "command",
            "commandId": "cmd-center-test",
            "sequence": 1,
            "action": "start_landing",
            "context": {"message": "Manual center homography"},
            "timestamp": time.time(),
        }
        await websocket.send(json.dumps(command))
        print("start_landing 명령 전송 완료.")

        async def sender() -> None:
            while True:
                payload = {
                    "type": "homography",
                    "matrix": matrix_inv_list,
                    "timestamp": time.time(),
                }
                await websocket.send(json.dumps(payload))
                await asyncio.sleep(interval)

        async def receiver() -> None:
            async for raw in websocket:
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    print("수신 디코드 실패:", raw)
                    continue
                msg_type = message.get("type")
                if msg_type == "ack":
                    print(
                        f"ACK 수신 | action={message.get('action')} "
                        f"status={message.get('status')} "
                        f"commandId={message.get('commandId')}"
                    )
                else:
                    print("기타 메시지 수신:", message)

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        try:
            await asyncio.wait(
                {sender_task, receiver_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )
        finally:
            sender_task.cancel()
            receiver_task.cancel()
            await asyncio.gather(sender_task, receiver_task, return_exceptions=True)
            print("클라이언트 세션 종료:", websocket.remote_address)

    async with websockets.serve(
        handler,
        config.websocket.host,
        config.websocket.port,
    ):
        print(
            f"테스트 서버 실행 중: ws://{config.websocket.host}:{config.websocket.port} "
            "— 화면 중앙 기준 호모그래피를 주기적으로 전송합니다."
        )
        await asyncio.Future()


def main() -> None:
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        print("\n테스트 서버를 종료합니다.")


if __name__ == "__main__":
    main()
