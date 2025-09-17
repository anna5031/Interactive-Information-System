#!/usr/bin/env python3
"""
오디오 관리 모듈 - 스피커 및 출력 디바이스 관리
"""

import os
import subprocess
import tempfile
import logging
import pygame
from gtts import gTTS
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class AudioManager:
    def __init__(self):
        self.usb_sink_name = None
        self.original_sink = None

        # pygame mixer 초기화
        try:
            pygame.mixer.quit()  # 기존 mixer 종료
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            logger.info("pygame mixer 초기화 완료")
        except pygame.error as e:
            logger.warning(f"pygame mixer 초기화 실패: {e}")

    def get_pulseaudio_sinks(self) -> List[str]:
        """PulseAudio 출력 디바이스 목록 가져오기"""
        try:
            result = subprocess.run(['pactl', 'list', 'short', 'sinks'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
        except Exception as e:
            logger.error(f"PulseAudio sink 목록 가져오기 실패: {e}")
        return []

    def get_current_default_sink(self) -> Optional[str]:
        """현재 기본 출력 디바이스 확인"""
        try:
            result = subprocess.run(['pactl', 'info'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'Default Sink:' in line:
                    return line.split('Default Sink:')[1].strip()
        except Exception as e:
            logger.error(f"기본 sink 확인 실패: {e}")
        return None

    def find_usb_audio_devices(self) -> Tuple[List[str], List[str]]:
        """USB 오디오 디바이스 찾기 (우선순위 기반)"""
        sinks = self.get_pulseaudio_sinks()
        usb_sinks = []
        hdmi_sinks = []

        # USB 디바이스 우선순위 키워드
        usb_keywords = [
            'usb',           # 일반적인 USB 키워드
            'uac',           # USB Audio Class
            'jieli',         # Jieli Technology
            'demo',          # UACDemo
            'audio',         # Audio 디바이스
            'headset',       # USB 헤드셋
            'speaker'        # USB 스피커
        ]

        # HDMI 키워드
        hdmi_keywords = ['hdmi', 'displayport', 'dp']

        for sink in sinks:
            if sink.strip():
                try:
                    sink_parts = sink.split('\t')
                    if len(sink_parts) >= 2:
                        sink_name = sink_parts[1]
                        sink_lower = sink_name.lower()

                        # USB 디바이스 검사
                        if any(keyword in sink_lower for keyword in usb_keywords):
                            usb_sinks.append(sink_name)
                            logger.info(f"USB 오디오 디바이스 발견: {sink_name}")

                        # HDMI 디바이스 검사
                        elif any(keyword in sink_lower for keyword in hdmi_keywords):
                            hdmi_sinks.append(sink_name)
                            logger.info(f"HDMI 오디오 디바이스 발견: {sink_name}")

                except Exception as e:
                    logger.warning(f"sink 분석 중 오류: {e}")

        return usb_sinks, hdmi_sinks

    def set_audio_output(self, sink_name: str) -> bool:
        """오디오 출력 디바이스 설정"""
        try:
            # 현재 기본 디바이스 저장
            if not self.original_sink:
                self.original_sink = self.get_current_default_sink()
                logger.info(f"현재 기본 출력 저장: {self.original_sink}")

            # 새 디바이스를 기본으로 설정
            result = subprocess.run(['pactl', 'set-default-sink', sink_name],
                                    capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"오디오 출력 설정 완료: {sink_name}")
                return True
            else:
                logger.error(f"오디오 출력 설정 실패: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"오디오 출력 설정 중 오류: {e}")
            return False

    def setup_preferred_audio_output(self) -> bool:
        """선호하는 오디오 출력 설정 (USB 우선, 없으면 HDMI 차단)"""
        usb_sinks, hdmi_sinks = self.find_usb_audio_devices()

        # USB 디바이스가 있으면 첫 번째 것을 사용
        if usb_sinks:
            preferred_sink = usb_sinks[0]
            logger.info(f"USB 오디오 디바이스 사용: {preferred_sink}")
            if self.set_audio_output(preferred_sink):
                self.usb_sink_name = preferred_sink
                return True

        # USB가 없고 HDMI만 있는 경우 경고
        elif hdmi_sinks:
            logger.warning("USB 오디오 디바이스를 찾을 수 없습니다. HDMI 출력이 감지되었습니다.")
            logger.warning("USB 스피커를 연결하고 다시 시도하세요.")

        else:
            logger.warning("USB 또는 HDMI 오디오 디바이스를 찾을 수 없습니다.")
            logger.info("기본 오디오 출력을 사용합니다.")

        return False

    def test_audio_playback(self, test_text: str = "오디오 테스트입니다.") -> bool:
        """오디오 재생 테스트"""
        try:
            # TTS로 테스트 오디오 생성
            tts = gTTS(text=test_text, lang='ko')

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                mp3_file = temp_file.name
                tts.save(mp3_file)

                # WAV 변환 (호환성 향상)
                wav_file = mp3_file.replace('.mp3', '.wav')
                conversion_result = os.system(
                    f"ffmpeg -i '{mp3_file}' -acodec pcm_s16le -ar 48000 '{wav_file}' -y 2>/dev/null"
                )

                success = False

                # 여러 재생 방법 시도
                playback_methods = [
                    ("paplay WAV", f"paplay '{wav_file}'"),
                    ("paplay MP3", f"paplay '{mp3_file}'"),
                    ("pygame", None),  # pygame은 별도 처리
                ]

                if self.usb_sink_name:
                    playback_methods.insert(0,
                        ("paplay USB", f"paplay --device={self.usb_sink_name} '{wav_file}'")
                    )

                for method_name, command in playback_methods:
                    try:
                        if method_name == "pygame":
                            # pygame으로 재생
                            pygame.mixer.music.load(mp3_file)
                            pygame.mixer.music.play()

                            # 재생 완료 대기
                            while pygame.mixer.music.get_busy():
                                pygame.time.wait(100)

                            success = True
                            logger.info(f"오디오 재생 성공: {method_name}")
                            break
                        else:
                            # 시스템 명령어로 재생
                            result = os.system(f"{command} 2>/dev/null")
                            if result == 0:
                                success = True
                                logger.info(f"오디오 재생 성공: {method_name}")
                                break
                    except Exception as e:
                        logger.warning(f"{method_name} 재생 실패: {e}")

                # 임시 파일 정리
                for file_path in [mp3_file, wav_file]:
                    if os.path.exists(file_path):
                        os.unlink(file_path)

                return success

        except Exception as e:
            logger.error(f"오디오 재생 테스트 중 오류: {e}")
            return False

    def text_to_speech_and_play(self, text: str) -> bool:
        """텍스트를 음성으로 변환하여 재생"""
        try:
            tts = gTTS(text=text, lang='ko')

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                mp3_file = temp_file.name
                tts.save(mp3_file)

                success = False

                # 재생 방법 시도 (우선순위 순)
                try:
                    # 방법 1: pygame으로 재생
                    pygame.mixer.music.load(mp3_file)
                    pygame.mixer.music.play()

                    # 재생 완료 대기
                    while pygame.mixer.music.get_busy():
                        pygame.time.wait(100)

                    success = True

                except Exception as e:
                    logger.warning(f"pygame 재생 실패: {e}")

                    # 방법 2: paplay로 재생
                    try:
                        if self.usb_sink_name:
                            result = os.system(f"paplay --device={self.usb_sink_name} '{mp3_file}' 2>/dev/null")
                        else:
                            result = os.system(f"paplay '{mp3_file}' 2>/dev/null")

                        if result == 0:
                            success = True
                    except Exception as e2:
                        logger.warning(f"paplay 재생 실패: {e2}")

                # 임시 파일 삭제
                if os.path.exists(mp3_file):
                    os.unlink(mp3_file)

                return success

        except Exception as e:
            logger.error(f"TTS 재생 중 오류: {e}")
            return False

    def restore_original_audio(self):
        """원래 오디오 설정으로 복원"""
        if self.original_sink:
            try:
                subprocess.run(['pactl', 'set-default-sink', self.original_sink])
                logger.info(f"오디오 출력을 원래대로 복원: {self.original_sink}")
            except Exception as e:
                logger.warning(f"오디오 설정 복원 실패: {e}")

    def cleanup(self):
        """리소스 정리"""
        try:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
        except Exception as e:
            logger.warning(f"pygame mixer 정리 중 오류: {e}")