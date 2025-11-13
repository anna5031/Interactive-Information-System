import logging
from typing import List, Optional

import sounddevice as sd  # type: ignore

from .arduino import check_arduino
from .camera import check_camera
from .config import DevicePreferences
from .microphone import check_microphone
from .speaker import check_speaker
from .types import DeviceCheckResult

logger = logging.getLogger(__name__)


class DeviceManager:
    def __init__(
        self,
        preferences: Optional[DevicePreferences] = None,
        use_dummy_arduino: bool = False,
    ) -> None:
        self.preferences = preferences or DevicePreferences()
        self.use_dummy_arduino = use_dummy_arduino
        self.results: List[DeviceCheckResult] = []
        self.selected_input: Optional[tuple[int, str]] = None
        self.selected_output: Optional[tuple[int, str]] = None

    def initialize(self) -> None:
        logger.info("장치 초기화 루틴 시작")
        self.results = [
            check_microphone(),
            check_speaker(),
            check_camera(
                source=self.preferences.camera_source,
                frame_size=self.preferences.camera_frame_size,
            ),
        ]

        if self.use_dummy_arduino:
            logger.info("아두이노 확인을 건너뜁니다 (USE_DUMMY_ARDUINO=True).")
        else:
            self.results.append(check_arduino())

        failed = [result for result in self.results if not result.ok]
        for result in self.results:
            status = "OK" if result.ok else "FAILED"
            logger.info(" - %s: %s", result.name, status)
            if result.detail:
                logger.info("   detail: %s", result.detail)

        if failed:
            details = ", ".join(
                f"{res.name}: {res.detail or '알 수 없음'}" for res in failed
            )
            raise RuntimeError(f"장치 초기화 실패: {details}")

        self._apply_audio_preferences()
        logger.info("장치 초기화 루틴 완료")

    def shutdown(self) -> None:
        logger.info("장치 종료 루틴 시작")
        logger.info("장치 종료 루틴 완료")

    def _apply_audio_preferences(self) -> None:
        mic_result = self._find_result("마이크")
        spk_result = self._find_result("스피커")

        input_index = self._select_audio_device(
            mic_result,
            names=self.preferences.microphone_priority_names,
        )
        output_index = self._select_audio_device(
            spk_result,
            names=self.preferences.speaker_priority_names,
        )

        if input_index is None:
            logger.warning("입력 장치를 선택하지 못했습니다.")
            return
        if output_index is None:
            logger.warning("출력 장치를 선택하지 못했습니다.")
            return

        input_name = self._resolve_audio_name(mic_result, input_index)
        output_name = self._resolve_audio_name(spk_result, output_index)

        sd.default.device = (input_index, output_index)
        self.selected_input = (input_index, input_name)
        self.selected_output = (output_index, output_name)
        logger.info(
            "사운드 기본 장치 설정 완료 (입력: %s - %s, 출력: %s - %s)",
            input_index,
            input_name,
            output_index,
            output_name,
        )

    def _select_audio_device(
        self,
        result: Optional[DeviceCheckResult],
        *,
        names: List[str],
    ) -> Optional[int]:
        if result is None or not result.meta:
            return None

        devices = result.meta.get("devices", [])
        if not devices:
            return None

        for priority_name in names:
            for device in devices:
                if priority_name.lower() in device["name"].lower():
                    logger.info(
                        "%s 우선순위 이름 선택: '%s' → %s (%s)",
                        result.name,
                        priority_name,
                        device["index"],
                        device["name"],
                    )
                    return device["index"]
            logger.warning(
                "%s 우선순위 이름을 찾지 못했습니다: %s",
                result.name,
                priority_name,
            )

        return devices[0]["index"]

    def _find_result(self, name: str) -> Optional[DeviceCheckResult]:
        for result in self.results:
            if result.name == name:
                return result
        return None

    @staticmethod
    def _resolve_audio_name(
        result: Optional[DeviceCheckResult], device_index: int
    ) -> str:
        if result is None or not result.meta:
            return "unknown"
        for device in result.meta.get("devices", []):
            if device["index"] == device_index:
                return device["name"]
        return "unknown"
