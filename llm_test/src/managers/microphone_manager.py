"""구성 설정을 기반으로 동작하는 마이크 캡처 관리자."""

import copy
import logging
import os
import subprocess
import sys
import wave
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config")
if CONFIG_PATH not in sys.path:
    sys.path.insert(0, CONFIG_PATH)
try:
    from microphone_config import MICROPHONE_CONFIG as USER_MICROPHONE_CONFIG
except ImportError as exc:
    raise RuntimeError("config/microphone_config.py 를 불러올 수 없습니다.") from exc


def _load_microphone_config() -> dict:
    """마이크 설정을 안전하게 복사해 반환."""
    if not isinstance(USER_MICROPHONE_CONFIG, dict):
        raise RuntimeError(
            "microphone_config.MICROPHONE_CONFIG 형식이 올바르지 않습니다."
        )

    return copy.deepcopy(USER_MICROPHONE_CONFIG)


class MicrophoneManager:
    def __init__(self):
        self.mic_config = _load_microphone_config()

        audio_settings = self._load_audio_settings()

        self.CHUNK = audio_settings["chunk_size"]
        self.DTYPE = audio_settings["dtype"]
        self.CHANNELS = audio_settings["channels"]
        self.RATE = audio_settings["sample_rate"]
        self.SILENCE_THRESHOLD = audio_settings["silence_threshold"]
        self.SILENCE_DURATION = audio_settings["silence_duration"]

        # 디바이스 인덱스
        self.input_device_index = None
        self.preferred_device_name = None

        # sounddevice 기본값을 설정
        sd.default.samplerate = self.RATE
        sd.default.channels = self.CHANNELS
        sd.default.dtype = self.DTYPE

        # 상태 변수
        self.is_listening = False

        selection_mode_value = self.mic_config.get("selection_mode")
        if selection_mode_value is None:
            raise RuntimeError(
                "microphone_config 설정에 'selection_mode' 키가 없습니다."
            )
        if (
            not isinstance(selection_mode_value, str)
            or not selection_mode_value.strip()
        ):
            raise RuntimeError(
                "microphone_config.selection_mode 값이 올바르지 않습니다."
            )

        self.selection_mode = selection_mode_value.strip().lower()

        logger.info("Microphone init complete (selection_mode=%s)", self.selection_mode)

    def get_audio_devices(self) -> Tuple[List[Tuple[int, str]], Optional[int]]:
        """사용 가능한 입력 디바이스 목록 및 추천 디바이스 반환"""
        input_devices = []
        recommended_input = None

        devices = sd.query_devices()
        for i, device in enumerate(devices):
            try:
                if device["max_input_channels"] > 0:
                    device_name = device["name"]
                    input_devices.append((i, device_name))

                    logger.info(f"입력 디바이스 {i}: {device_name}")
                    logger.info(f"  - 최대 입력 채널: {device['max_input_channels']}")
                    logger.info(f"  - 기본 샘플레이트: {device['default_samplerate']}")

            except Exception as e:
                logger.warning(f"디바이스 {i} 정보 읽기 실패: {e}")

        self.preferred_device_name = None
        recommended_input = self._choose_recommended_device(input_devices)

        return input_devices, recommended_input

    def _choose_recommended_device(
        self, input_devices: List[Tuple[int, str]]
    ) -> Optional[int]:
        """설정값에 맞춰 추천 디바이스를 결정."""
        if not input_devices:
            return None

        selection_mode = self.selection_mode

        if selection_mode == "priority":
            priority_idx = self._select_device_from_priority(input_devices)
            if priority_idx is not None:
                return priority_idx

            if not self.mic_config.get("fallback_to_auto", True):
                logger.warning(
                    "priority 설정과 일치하는 마이크가 없어 선택을 중단합니다."
                )
                return None

        auto_idx = self._select_device_from_auto(input_devices)
        if auto_idx is not None:
            return auto_idx

        first_idx, first_name = input_devices[0]
        self.preferred_device_name = first_name
        logger.info(f"첫 번째 마이크 선택: {first_name} (인덱스: {first_idx})")
        return first_idx

    def _select_device_from_priority(
        self, input_devices: List[Tuple[int, str]]
    ) -> Optional[int]:
        """우선순위 설정에 따라 디바이스를 선택."""
        name_patterns = self.mic_config.get("priority_device_names", []) or []

        for pattern in name_patterns:
            if not isinstance(pattern, str):
                continue

            lowered_pattern = pattern.lower()

            for idx, name in input_devices:
                if lowered_pattern in name.lower():
                    self.preferred_device_name = name
                    logger.info(
                        f"priority 이름 '{pattern}' 일치: {name} (인덱스: {idx})"
                    )
                    return idx

        return None

    def _select_device_from_auto(
        self, input_devices: List[Tuple[int, str]]
    ) -> Optional[int]:
        """자동 선택 규칙에 따라 디바이스를 선택."""
        if "auto_priority" not in self.mic_config:
            raise RuntimeError(
                "microphone_config 설정에 'auto_priority' 키가 없습니다."
            )

        priorities = self.mic_config["auto_priority"]
        if not isinstance(priorities, (list, tuple)):
            raise RuntimeError(
                "microphone_config.auto_priority 값이 올바르지 않습니다."
            )

        for priority in priorities:
            if isinstance(priority, dict):
                keywords = priority.get("keywords", []) or []
                label = priority.get("label")
            elif isinstance(priority, (list, tuple, set)):
                keywords = list(priority)
                label = None
            elif isinstance(priority, str):
                keywords = [priority]
                label = None
            else:
                continue

            lowered_keywords = [kw.lower() for kw in keywords if isinstance(kw, str)]
            if not lowered_keywords:
                continue

            for idx, name in input_devices:
                name_lower = name.lower()
                if any(keyword in name_lower for keyword in lowered_keywords):
                    self.preferred_device_name = name
                    description = label or f"키워드 {', '.join(lowered_keywords)}"
                    logger.info(
                        f"{description} 기준으로 마이크 선택: {name} (인덱스: {idx})"
                    )
                    return idx

        return None

    def setup_microphone(self, device_index: Optional[int] = None) -> bool:
        """마이크 설정"""
        if device_index is None:
            # 1. PulseAudio USB 마이크를 기본으로 설정 시도
            usb_source = self.find_usb_microphone()
            if usb_source:
                self.set_usb_microphone_as_default(usb_source)

            # 2. PyAudio 디바이스 자동 선택
            _, recommended_input = self.get_audio_devices()
            device_index = recommended_input

        if device_index is not None:
            self.input_device_index = device_index
            logger.info(f"마이크 설정 완료 - 디바이스 인덱스: {device_index}")
            return True
        else:
            logger.error("사용 가능한 마이크를 찾을 수 없습니다.")
            return False

    def test_microphone(self, duration: float) -> bool:
        """마이크 테스트 (지정한 시간만큼 녹음하여 음성 활동 감지)"""
        logger.info(f"마이크 테스트 시작 ({duration}초간 녹음)")

        # 여러 디바이스로 테스트 시도
        test_devices = []

        # 현재 설정된 디바이스
        if self.input_device_index is not None:
            test_devices.append(("현재 설정", self.input_device_index))

        # USB 마이크 찾기 (PyAudio 기준)
        input_devices, _ = self.get_audio_devices()
        for idx, name in input_devices:
            name_lower = name.lower()
            if any(
                keyword in name_lower
                for keyword in ["usb", "pcm2902", "c-media", "texas"]
            ):
                test_devices.append((f"USB: {name}", idx))

        # 기본 디바이스들
        for idx, name in input_devices:
            if any(keyword in name.lower() for keyword in ["pulse", "default"]):
                test_devices.append((f"기본: {name}", idx))
                break

        # 각 디바이스로 테스트
        for device_desc, device_idx in test_devices:
            logger.info(f"테스트 중: {device_desc}")
            success = self._test_single_device(device_idx, duration)
            if success:
                # 성공한 디바이스로 설정 업데이트
                self.input_device_index = device_idx
                logger.info(f"✅ 마이크 테스트 성공! 디바이스 변경: {device_desc}")
                return True

        logger.warning("⚠️ 모든 마이크 테스트 실패")
        return False

    def _test_single_device(self, device_index: Optional[int], duration: float) -> bool:
        """단일 디바이스로 마이크 테스트"""
        try:
            # 스트림 설정
            device = device_index if device_index is not None else None

            # sounddevice 스트림 생성
            stream = sd.InputStream(
                device=device,
                channels=self.CHANNELS,
                samplerate=self.RATE,
                dtype=self.DTYPE,
                blocksize=self.CHUNK,
            )
            stream.start()

        except Exception as e:
            logger.warning(f"디바이스 {device_index} 스트림 열기 실패: {e}")
            return False

        frames = []
        chunks_to_record = int(self.RATE / self.CHUNK * duration)
        max_amplitude = 0
        avg_amplitude = 0
        frames_recorded = 0

        try:
            for i in range(chunks_to_record):
                try:
                    audio_chunk, overflowed = stream.read(self.CHUNK)
                    if overflowed:
                        logger.warning("오디오 버퍼 오버플로우")

                    audio_chunk = audio_chunk.flatten().astype(np.int16)
                    frames.append(audio_chunk.tobytes())

                    # 진폭 분석
                    chunk_max = np.max(np.abs(audio_chunk))
                    chunk_avg = np.mean(np.abs(audio_chunk))

                    max_amplitude = max(max_amplitude, chunk_max)
                    avg_amplitude += chunk_avg
                    frames_recorded += 1

                    # 실시간 피드백 (매 초마다)
                    if i % (self.RATE // self.CHUNK) == 0 and i > 0:
                        logger.info(
                            f"  진행: {i // (self.RATE // self.CHUNK)}초 | 최대 진폭: {max_amplitude}"
                        )

                except Exception as e:
                    logger.warning(f"청크 읽기 오류: {e}")
                    break

        finally:
            stream.stop()
            stream.close()

        if frames_recorded > 0:
            avg_amplitude = avg_amplitude / frames_recorded

        logger.info(f"  결과 - 최대: {max_amplitude}, 평균: {avg_amplitude:.1f}")

        # 더 관대한 기준으로 판정
        if max_amplitude > 50:  # 기준을 낮춤
            logger.info(f"  ✅ 음성 신호 감지됨!")
            return True
        else:
            logger.info(f"  ❌ 음성 신호 없음")
            return False

    def detect_silence(self, audio_data: np.ndarray) -> bool:
        """음성 활동 감지 (VAD)"""
        if len(audio_data) == 0:
            return True

        # RMS 계산 (안전한 방식)
        audio_data = audio_data.astype(np.float64)
        mean_square = np.mean(audio_data**2)

        if mean_square <= 0:
            return True

        rms = np.sqrt(mean_square)
        return rms < self.SILENCE_THRESHOLD

    def record_audio(self) -> Optional[bytes]:
        """음성 녹음 (VAD 적용)"""
        logger.info("🎤 음성 인식 대기 중... (말씀하세요)")

        try:
            # 입력 디바이스 설정
            device = (
                self.input_device_index if self.input_device_index is not None else None
            )

            stream = sd.InputStream(
                device=device,
                channels=self.CHANNELS,
                samplerate=self.RATE,
                dtype=self.DTYPE,
                blocksize=self.CHUNK,
            )
            stream.start()

        except Exception as e:
            logger.error(f"오디오 스트림 열기 실패: {e}")
            return None

        frames = []
        silent_chunks = 0
        recording_started = False

        try:
            while self.is_listening:
                try:
                    audio_chunk, overflowed = stream.read(self.CHUNK)
                    if overflowed:
                        logger.warning("오디오 버퍼 오버플로우")

                    audio_chunk = audio_chunk.flatten().astype(np.int16)

                    is_silent = self.detect_silence(audio_chunk)

                    if not is_silent:
                        if not recording_started:
                            logger.info("🔴 녹음 시작")
                            recording_started = True
                        frames.append(audio_chunk.tobytes())
                        silent_chunks = 0
                    elif recording_started:
                        frames.append(audio_chunk.tobytes())
                        silent_chunks += 1

                        # 무음이 지속되면 녹음 종료
                        if silent_chunks > (
                            self.SILENCE_DURATION * self.RATE / self.CHUNK
                        ):
                            logger.info("⏹️  녹음 완료")
                            break

                except Exception as e:
                    logger.warning(f"오디오 청크 읽기 오류: {e}")
                    break

        except Exception as e:
            logger.error(f"녹음 중 오류: {e}")
        finally:
            stream.stop()
            stream.close()

        if frames:
            return b"".join(frames)
        return None

    def save_audio_to_wav(self, audio_data: bytes, file_path: str):
        """오디오 데이터를 WAV 파일로 저장"""
        try:
            with wave.open(file_path, "wb") as wav_file:
                wav_file.setnchannels(self.CHANNELS)
                wav_file.setsampwidth(2)  # int16 샘플은 2바이트
                wav_file.setframerate(self.RATE)
                wav_file.writeframes(audio_data)
            logger.info(f"오디오 저장 완료: {file_path}")
        except Exception as e:
            logger.error(f"오디오 저장 실패: {e}")

    def start_listening(self):
        """음성 인식 시작"""
        self.is_listening = True
        logger.info("음성 인식 시작")

    def stop_listening(self):
        """음성 인식 중지"""
        self.is_listening = False
        logger.info("음성 인식 중지")

    def get_pulseaudio_sources(self) -> List[str]:
        """PulseAudio 입력 소스 목록 가져오기"""
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sources"], capture_output=True, text=True
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")
        except Exception as e:
            logger.error(f"PulseAudio source 목록 가져오기 실패: {e}")
        return []

    def find_usb_microphone(self) -> Optional[str]:
        """USB 마이크 찾기 (PulseAudio 기반)"""
        sources = self.get_pulseaudio_sources()

        # 더 구체적인 USB 마이크 키워드
        usb_keywords = [
            "usb",
            "pcm2902",
            "c-media",
            "texas",
            "microphone",
            "mic",
            "headset",
            "webcam",
            "uac",
        ]

        for source in sources:
            if source.strip():
                try:
                    source_parts = source.split("\t")
                    if len(source_parts) >= 2:
                        source_name = source_parts[1]
                        source_lower = source_name.lower()

                        # USB 관련 키워드 검색
                        if any(keyword in source_lower for keyword in usb_keywords):
                            # 모니터 소스는 제외 (출력의 녹음용)
                            if "monitor" not in source_lower:
                                logger.info(f"USB 마이크 발견: {source_name}")
                                return source_name
                except Exception as e:
                    logger.warning(f"source 분석 중 오류: {e}")

        logger.warning("USB 마이크를 찾지 못했습니다.")
        return None

    def set_usb_microphone_as_default(self, source_name: str) -> bool:
        """USB 마이크를 기본 입력으로 설정"""
        try:
            result = subprocess.run(
                ["pactl", "set-default-source", source_name],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info(f"USB 마이크를 기본 입력으로 설정: {source_name}")
                return True
            else:
                logger.error(f"USB 마이크 설정 실패: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"USB 마이크 설정 중 오류: {e}")
            return False

    def cleanup(self):
        """리소스 정리"""
        try:
            self.stop_listening()
            # sounddevice는 별도의 terminate 불필요
        except Exception as e:
            logger.warning(f"마이크 관리자 리소스 정리 중 오류: {e}")
        logger.info("마이크 관리자 리소스 정리 완료")

    def _load_audio_settings(self) -> dict:
        audio = self.mic_config.get("audio")
        if audio is None or not isinstance(audio, dict):
            raise RuntimeError(
                "microphone_config 설정에 'audio' 키가 없거나 형식이 올바르지 않습니다."
            )

        required = {
            "chunk_size": int,
            "dtype": str,
            "channels": int,
            "sample_rate": int,
            "silence_threshold": (int, float),
            "silence_duration": (int, float),
        }

        missing = [key for key in required if key not in audio]
        if missing:
            raise RuntimeError(
                f"microphone_config.audio 설정에 누락된 키가 있습니다: {', '.join(sorted(missing))}"
            )

        settings = {}
        for key, expected_type in required.items():
            value = audio[key]
            if not isinstance(value, expected_type):
                raise RuntimeError(
                    f"microphone_config.audio.{key} 값이 올바르지 않습니다."
                )
            settings[key] = value

        dtype_map = {"int16": np.int16, "float32": np.float32, "float64": np.float64}

        dtype_key = settings["dtype"].lower()
        if dtype_key not in dtype_map:
            raise RuntimeError(
                "microphone_config.audio.dtype 값이 지원되지 않습니다 (int16/float32/float64)."
            )

        settings["dtype"] = dtype_map[dtype_key]

        settings["chunk_size"] = int(settings["chunk_size"])
        settings["channels"] = int(settings["channels"])
        settings["sample_rate"] = int(settings["sample_rate"])
        settings["silence_threshold"] = float(settings["silence_threshold"])
        settings["silence_duration"] = float(settings["silence_duration"])

        return settings
