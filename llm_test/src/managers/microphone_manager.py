#!/usr/bin/env python3
"""
마이크 관리 모듈 - 마이크 입력 및 녹음 관리
"""

import sounddevice as sd
import numpy as np
import wave
import tempfile
import logging
import subprocess
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class MicrophoneManager:
    def __init__(self):
        # 오디오 설정
        self.CHUNK = 1024
        self.DTYPE = np.int16
        self.CHANNELS = 1
        self.RATE = 16000
        self.SILENCE_THRESHOLD = 500
        self.SILENCE_DURATION = 2.0

        # 디바이스 인덱스
        self.input_device_index = None
        self.preferred_device_name = None

        # SoundDevice 설정
        sd.default.samplerate = self.RATE
        sd.default.channels = self.CHANNELS
        sd.default.dtype = self.DTYPE

        # 상태 변수
        self.is_listening = False

        logger.info("마이크 관리자 초기화 완료")

    def get_audio_devices(self) -> Tuple[List[Tuple[int, str]], Optional[int]]:
        """사용 가능한 입력 디바이스 목록 및 추천 디바이스 반환"""
        input_devices = []
        recommended_input = None

        devices = sd.query_devices()
        for i, device in enumerate(devices):
            try:
                if device['max_input_channels'] > 0:
                    device_name = device['name']
                    input_devices.append((i, device_name))

                    logger.info(f"입력 디바이스 {i}: {device_name}")
                    logger.info(f"  - 최대 입력 채널: {device['max_input_channels']}")
                    logger.info(f"  - 기본 샘플레이트: {device['default_samplerate']}")

            except Exception as e:
                logger.warning(f"디바이스 {i} 정보 읽기 실패: {e}")

        # 추천 디바이스 선택 (우선순위)
        usb_keywords = ['usb', 'microphone', 'mic', 'headset', 'webcam']
        pulse_keywords = ['pulse', 'default']

        # 1순위: USB 마이크
        for idx, name in input_devices:
            if any(keyword in name.lower() for keyword in usb_keywords):
                recommended_input = idx
                self.preferred_device_name = name
                logger.info(f"USB 마이크 발견: {name} (인덱스: {idx})")
                break

        # 2순위: pulse 또는 default
        if not recommended_input:
            for idx, name in input_devices:
                if any(keyword in name.lower() for keyword in pulse_keywords):
                    recommended_input = idx
                    self.preferred_device_name = name
                    logger.info(f"기본 마이크 선택: {name} (인덱스: {idx})")
                    break

        # 3순위: 첫 번째 사용 가능한 디바이스
        if not recommended_input and input_devices:
            recommended_input = input_devices[0][0]
            self.preferred_device_name = input_devices[0][1]
            logger.info(f"첫 번째 마이크 선택: {self.preferred_device_name} (인덱스: {recommended_input})")

        return input_devices, recommended_input

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

    def test_microphone(self, duration: float = 3.0) -> bool:
        """마이크 테스트 (3초간 녹음하여 음성 활동 감지)"""
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
            if any(keyword in name_lower for keyword in ['usb', 'pcm2902', 'c-media', 'texas']):
                test_devices.append((f"USB: {name}", idx))

        # 기본 디바이스들
        for idx, name in input_devices:
            if any(keyword in name.lower() for keyword in ['pulse', 'default']):
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
                blocksize=self.CHUNK
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
                        logger.info(f"  진행: {i // (self.RATE // self.CHUNK)}초 | 최대 진폭: {max_amplitude}")

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
        mean_square = np.mean(audio_data ** 2)

        if mean_square <= 0:
            return True

        rms = np.sqrt(mean_square)
        return rms < self.SILENCE_THRESHOLD

    def record_audio(self) -> Optional[bytes]:
        """음성 녹음 (VAD 적용)"""
        logger.info("🎤 음성 인식 대기 중... (말씀하세요)")

        try:
            # 입력 디바이스 설정
            device = self.input_device_index if self.input_device_index is not None else None

            stream = sd.InputStream(
                device=device,
                channels=self.CHANNELS,
                samplerate=self.RATE,
                dtype=self.DTYPE,
                blocksize=self.CHUNK
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
                        if silent_chunks > (self.SILENCE_DURATION * self.RATE / self.CHUNK):
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
            return b''.join(frames)
        return None

    def save_audio_to_wav(self, audio_data: bytes, file_path: str):
        """오디오 데이터를 WAV 파일로 저장"""
        try:
            with wave.open(file_path, 'wb') as wav_file:
                wav_file.setnchannels(self.CHANNELS)
                wav_file.setsampwidth(2)  # int16 = 2 bytes
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
            result = subprocess.run(['pactl', 'list', 'short', 'sources'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
        except Exception as e:
            logger.error(f"PulseAudio source 목록 가져오기 실패: {e}")
        return []

    def find_usb_microphone(self) -> Optional[str]:
        """USB 마이크 찾기 (PulseAudio 기반)"""
        sources = self.get_pulseaudio_sources()

        # 더 구체적인 USB 마이크 키워드
        usb_keywords = ['usb', 'pcm2902', 'c-media', 'texas', 'microphone', 'mic', 'headset', 'webcam', 'uac']

        for source in sources:
            if source.strip():
                try:
                    source_parts = source.split('\t')
                    if len(source_parts) >= 2:
                        source_name = source_parts[1]
                        source_lower = source_name.lower()

                        # USB 관련 키워드 검색
                        if any(keyword in source_lower for keyword in usb_keywords):
                            # 모니터 소스는 제외 (출력의 녹음용)
                            if 'monitor' not in source_lower:
                                logger.info(f"USB 마이크 발견: {source_name}")
                                return source_name
                except Exception as e:
                    logger.warning(f"source 분석 중 오류: {e}")

        logger.warning("USB 마이크를 찾지 못했습니다.")
        return None

    def set_usb_microphone_as_default(self, source_name: str) -> bool:
        """USB 마이크를 기본 입력으로 설정"""
        try:
            result = subprocess.run(['pactl', 'set-default-source', source_name],
                                  capture_output=True, text=True)
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