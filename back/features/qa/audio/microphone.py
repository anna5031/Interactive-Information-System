"""Configuration-driven microphone capture manager."""

from __future__ import annotations

import copy
import logging
import os
import subprocess
import wave
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

from config.qa.microphone_config import (
    MICROPHONE_CONFIG as USER_MICROPHONE_CONFIG,
)  # type: ignore


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
        self.target_sample_rate = audio_settings["sample_rate"]
        self.RATE = self.target_sample_rate  # 외부에 공개되는 기본 샘플레이트
        self._stream_samplerate = float(self.RATE)
        self._samplerate_fallback_active = False
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
            matched = False

            for idx, name in input_devices:
                if lowered_pattern in name.lower():
                    self.preferred_device_name = name
                    logger.info(
                        f"priority 이름 '{pattern}' 일치: {name} (인덱스: {idx})"
                    )
                    matched = True
                    return idx
            if not matched:
                logger.warning("priority 목록에서 '%s' 디바이스를 찾지 못했습니다.", pattern)

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
            stream, samplerate = self._open_input_stream(
                device_index, update_state=True, start_stream=False
            )
            if stream is None or samplerate is None:
                logger.error("선택한 마이크에서 사용할 수 있는 샘플레이트를 찾지 못했습니다.")
                return False
            try:
                stream.close()
            except Exception:
                pass
            logger.info(
                "마이크 설정 완료 - 디바이스 인덱스: %s (샘플레이트: %sHz)",
                device_index,
                samplerate,
            )
            return True

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

            stream, samplerate = self._open_input_stream(
                device, update_state=False, start_stream=True
            )
            if stream is None or samplerate is None:
                logger.warning(
                    f"디바이스 {device_index} 에서 사용할 수 있는 샘플레이트를 찾지 못했습니다."
                )
                return False

        except Exception as e:
            logger.warning(f"디바이스 {device_index} 스트림 열기 실패: {e}")
            return False

        frames = []
        chunks_to_record = int(samplerate / self.CHUNK * duration)
        if chunks_to_record <= 0:
            chunks_to_record = 1
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
                    interval = max(int(samplerate // self.CHUNK), 1)
                    if i % interval == 0 and i > 0:
                        logger.info(
                            f"  진행: {i // interval}초 | 최대 진폭: {max_amplitude}"
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
            self._apply_effective_samplerate(samplerate, reason="마이크 테스트")
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

            stream, samplerate = self._open_input_stream(
                device, update_state=True, start_stream=True
            )
            if stream is None or samplerate is None:
                logger.error("오디오 스트림에 사용할 수 있는 샘플레이트가 없습니다.")
                return None

        except Exception as e:
            logger.error(f"오디오 스트림 열기 실패: {e}")
            return None

        frames = []
        silent_chunks = 0
        recording_started = False
        silence_limit = max(
            int(self.SILENCE_DURATION * samplerate / self.CHUNK), 1
        )

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
                        if silent_chunks > silence_limit:
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
            audio_bytes = b"".join(frames)
            audio_bytes = self._resample_audio(audio_bytes, samplerate)
            return audio_bytes
        return None

    def _apply_effective_samplerate(self, samplerate: float, reason: str = "") -> None:
        """스트림 사용 샘플레이트를 갱신하고 상태를 기록."""
        try:
            samplerate_value = float(samplerate)
        except (TypeError, ValueError):
            logger.debug("잘못된 샘플레이트 값으로 갱신을 건너뜁니다: %s", samplerate)
            return

        previous_rate = self._stream_samplerate
        self._stream_samplerate = samplerate_value
        sd.default.samplerate = samplerate_value

        fallback_active = abs(samplerate_value - self.target_sample_rate) > 1e-3
        rate_changed = (
            abs(previous_rate - samplerate_value) > 1e-3
            if previous_rate is not None
            else True
        )

        self._samplerate_fallback_active = fallback_active

        if fallback_active and rate_changed:
            reason_suffix = f" ({reason})" if reason else ""
            logger.warning(
                "마이크 샘플레이트를 %sHz에서 %sHz로 변경합니다%s",
                int(previous_rate) if previous_rate else None,
                int(samplerate_value),
                reason_suffix,
            )
        elif rate_changed:
            logger.info(
                "마이크 샘플레이트를 %sHz에서 %sHz로 설정합니다",
                int(previous_rate) if previous_rate else None,
                int(samplerate_value),
            )

    def _get_samplerate_candidates(self, device_index: Optional[int]) -> List[float]:
        """해당 디바이스에서 시도할 샘플레이트 후보 목록을 구성."""
        candidates: List[float] = []

        if self.target_sample_rate:
            candidates.append(float(self.target_sample_rate))

        if self._stream_samplerate:
            candidates.append(float(self._stream_samplerate))

        device_info = None
        try:
            if device_index is not None:
                device_info = sd.query_devices(device_index)
            else:
                default_device = getattr(sd.default, "device", None)
                default_input = None
                if isinstance(default_device, (list, tuple)) and default_device:
                    default_input = default_device[0]
                elif isinstance(default_device, int):
                    default_input = default_device
                if isinstance(default_input, int) and default_input >= 0:
                    device_info = sd.query_devices(default_input)
        except Exception as exc:
            logger.debug("디바이스 정보 조회 실패 (device=%s): %s", device_index, exc)

        if isinstance(device_info, dict):
            default_rate = device_info.get("default_samplerate")
            try:
                if default_rate:
                    candidates.append(float(default_rate))
            except (TypeError, ValueError):
                pass

        candidates.extend([48000.0, 44100.0, 32000.0, 24000.0, 22050.0, 16000.0])

        ordered_unique: List[float] = []
        seen = set()
        for value in candidates:
            try:
                rate_val = float(value)
            except (TypeError, ValueError):
                continue
            if rate_val <= 0:
                continue
            key = round(rate_val, 3)
            if key in seen:
                continue
            seen.add(key)
            ordered_unique.append(rate_val)

        return ordered_unique

    def _open_input_stream(
        self,
        device_index: Optional[int],
        *,
        update_state: bool = True,
        start_stream: bool = True,
    ) -> Tuple[Optional[sd.InputStream], Optional[float]]:
        """샘플레이트 후보를 순회하며 입력 스트림을 생성."""
        candidates = self._get_samplerate_candidates(device_index)
        last_error: Optional[Exception] = None

        for samplerate in candidates:
            try:
                stream = sd.InputStream(
                    device=device_index if device_index is not None else None,
                    channels=self.CHANNELS,
                    samplerate=samplerate,
                    dtype=self.DTYPE,
                    blocksize=self.CHUNK,
                )
                if start_stream:
                    stream.start()

                if update_state:
                    reason = (
                        "디바이스 호환성"
                        if abs(samplerate - self.target_sample_rate) > 1e-3
                        else ""
                    )
                    self._apply_effective_samplerate(samplerate, reason=reason)

                return stream, samplerate

            except Exception as exc:
                last_error = exc
                logger.debug(
                    "샘플레이트 %sHz 스트림 생성 실패 (device=%s): %s",
                    int(samplerate)
                    if abs(samplerate - int(samplerate)) < 1e-6
                    else samplerate,
                    device_index,
                    exc,
                )

        if last_error:
            logger.error(
                "입력 스트림 생성 실패 (device=%s): %s",
                device_index,
                last_error,
            )
        else:
            logger.error(
                "입력 스트림 생성 실패: 시도할 샘플레이트가 없습니다 (device=%s).",
                device_index,
            )

        return None, None

    def _resample_audio(self, audio_bytes: bytes, source_rate: float) -> bytes:
        """장치 샘플레이트가 목표와 다를 때 오디오를 재샘플링."""
        if not audio_bytes:
            return audio_bytes

        try:
            source_rate_value = float(source_rate)
        except (TypeError, ValueError):
            logger.debug("재샘플링에 사용할 샘플레이트 값을 해석할 수 없습니다: %s", source_rate)
            return audio_bytes

        if (
            not self._samplerate_fallback_active
            or abs(source_rate_value - self.target_sample_rate) <= 1e-3
        ):
            return audio_bytes

        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        if samples.size == 0:
            return audio_bytes

        target_rate = float(self.target_sample_rate)
        if source_rate_value <= 0 or target_rate <= 0:
            return audio_bytes

        ratio = target_rate / source_rate_value
        target_length = max(int(round(samples.size * ratio)), 1)

        if target_length == samples.size:
            return audio_bytes

        source_positions = np.arange(samples.size, dtype=np.float64)
        target_positions = np.linspace(
            0, samples.size - 1, target_length, dtype=np.float64
        )

        resampled = np.interp(target_positions, source_positions, samples)
        resampled = np.clip(
            np.round(resampled),
            np.iinfo(np.int16).min,
            np.iinfo(np.int16).max,
        ).astype(np.int16)

        logger.info(
            "오디오를 재샘플링했습니다: %sHz -> %sHz (샘플 %s -> %s)",
            int(source_rate_value),
            int(target_rate),
            samples.size,
            resampled.size,
        )

        return resampled.tobytes()

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

    def get_stream_sample_rate(self) -> float:
        """현재 입력 스트림에서 사용하는 실제 샘플레이트."""
        return float(self._stream_samplerate)

    def has_samplerate_fallback(self) -> bool:
        """장치 호환성 때문에 샘플레이트를 변경했는지 여부."""
        return bool(self._samplerate_fallback_active)

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
