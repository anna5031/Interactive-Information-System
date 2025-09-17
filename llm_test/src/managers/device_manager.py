#!/usr/bin/env python3
"""
디바이스 관리 모듈 - USB 디바이스 안정적 인식 및 관리
"""

import subprocess
import logging
import time
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DeviceManager:
    def __init__(self):
        self.usb_devices = {}
        self.stable_devices = {}
        self.device_history = []

        # 디바이스 식별을 위한 키워드 맵
        self.audio_keywords = {
            'speaker': ['speaker', 'audio', 'sound', 'dac', 'headphone', 'jieli', 'uac'],
            'microphone': ['microphone', 'mic', 'webcam', 'camera', 'headset', 'input']
        }

        logger.info("디바이스 관리자 초기화 완료")

    def scan_usb_devices(self) -> Dict[str, dict]:
        """USB 디바이스 스캔"""
        devices = {}

        try:
            # lsusb 명령으로 USB 디바이스 목록 가져오기
            result = subprocess.run(['lsusb'], capture_output=True, text=True)

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        # lsusb 출력 파싱 (Bus XXX Device XXX: ID XXXX:XXXX Description)
                        match = re.match(r'Bus (\d+) Device (\d+): ID ([0-9a-f]{4}):([0-9a-f]{4}) (.+)', line)
                        if match:
                            bus, device, vendor_id, product_id, description = match.groups()

                            device_info = {
                                'bus': bus,
                                'device': device,
                                'vendor_id': vendor_id,
                                'product_id': product_id,
                                'description': description,
                                'device_id': f"{vendor_id}:{product_id}",
                                'timestamp': time.time()
                            }

                            # 오디오 관련 디바이스인지 확인
                            device_info['is_audio'] = self._is_audio_device(description)
                            device_info['device_type'] = self._classify_audio_device(description)

                            devices[device_info['device_id']] = device_info

        except Exception as e:
            logger.error(f"USB 디바이스 스캔 실패: {e}")

        return devices

    def _is_audio_device(self, description: str) -> bool:
        """디바이스가 오디오 관련인지 확인"""
        description_lower = description.lower()
        audio_keywords = ['audio', 'sound', 'speaker', 'microphone', 'mic', 'headset',
                         'webcam', 'camera', 'uac', 'headphone', 'dac']

        return any(keyword in description_lower for keyword in audio_keywords)

    def _classify_audio_device(self, description: str) -> Optional[str]:
        """오디오 디바이스 유형 분류"""
        description_lower = description.lower()

        # 스피커/출력 디바이스
        if any(keyword in description_lower for keyword in self.audio_keywords['speaker']):
            return 'speaker'

        # 마이크/입력 디바이스
        if any(keyword in description_lower for keyword in self.audio_keywords['microphone']):
            return 'microphone'

        # 일반 오디오 디바이스
        if self._is_audio_device(description):
            return 'audio'

        return None

    def get_stable_audio_devices(self, scan_count: int = 3, interval: float = 1.0) -> Dict[str, dict]:
        """안정적으로 인식되는 오디오 디바이스 찾기"""
        logger.info(f"안정적인 오디오 디바이스 검색 중... ({scan_count}회 스캔)")

        device_appearances = {}

        for i in range(scan_count):
            logger.info(f"스캔 {i+1}/{scan_count}")
            devices = self.scan_usb_devices()

            # 오디오 디바이스만 필터링
            audio_devices = {k: v for k, v in devices.items() if v['is_audio']}

            for device_id, device_info in audio_devices.items():
                if device_id not in device_appearances:
                    device_appearances[device_id] = []
                device_appearances[device_id].append(device_info)

            if i < scan_count - 1:
                time.sleep(interval)

        # 모든 스캔에서 발견된 디바이스만 안정적으로 간주
        stable_devices = {}
        for device_id, appearances in device_appearances.items():
            if len(appearances) == scan_count:  # 모든 스캔에서 발견됨
                # 가장 최근 정보 사용
                stable_devices[device_id] = appearances[-1]
                logger.info(f"안정적인 오디오 디바이스 발견: {appearances[-1]['description']}")

        self.stable_devices = stable_devices
        return stable_devices

    def get_pulseaudio_devices(self) -> Tuple[List[dict], List[dict]]:
        """PulseAudio 디바이스 목록 가져오기"""
        sinks = []  # 출력 디바이스
        sources = []  # 입력 디바이스

        try:
            # 출력 디바이스 (sinks)
            result = subprocess.run(['pactl', 'list', 'short', 'sinks'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            sinks.append({
                                'index': parts[0],
                                'name': parts[1],
                                'type': 'sink',
                                'is_usb': self._is_usb_pulse_device(parts[1])
                            })

            # 입력 디바이스 (sources)
            result = subprocess.run(['pactl', 'list', 'short', 'sources'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            sources.append({
                                'index': parts[0],
                                'name': parts[1],
                                'type': 'source',
                                'is_usb': self._is_usb_pulse_device(parts[1])
                            })

        except Exception as e:
            logger.error(f"PulseAudio 디바이스 목록 가져오기 실패: {e}")

        return sinks, sources

    def _is_usb_pulse_device(self, device_name: str) -> bool:
        """PulseAudio 디바이스명이 USB 디바이스인지 확인"""
        usb_indicators = ['usb', 'uac', 'jieli', 'demo']
        device_lower = device_name.lower()
        return any(indicator in device_lower for indicator in usb_indicators)

    def find_best_audio_devices(self) -> Tuple[Optional[str], Optional[str]]:
        """최적의 오디오 입출력 디바이스 찾기"""
        logger.info("최적의 오디오 디바이스 검색 중...")

        # 1. 안정적인 USB 디바이스 찾기
        stable_devices = self.get_stable_audio_devices()

        # 2. PulseAudio 디바이스 목록 가져오기
        sinks, sources = self.get_pulseaudio_devices()

        best_output = None
        best_input = None

        # 출력 디바이스 선택 (우선순위: USB > 기본)
        usb_sinks = [sink for sink in sinks if sink['is_usb']]
        if usb_sinks:
            best_output = usb_sinks[0]['name']
            logger.info(f"USB 출력 디바이스 선택: {best_output}")
        elif sinks:
            # USB가 없으면 HDMI가 아닌 첫 번째 디바이스 선택
            non_hdmi_sinks = [sink for sink in sinks if 'hdmi' not in sink['name'].lower()]
            if non_hdmi_sinks:
                best_output = non_hdmi_sinks[0]['name']
                logger.info(f"기본 출력 디바이스 선택: {best_output}")
            else:
                best_output = sinks[0]['name']
                logger.info(f"첫 번째 출력 디바이스 선택: {best_output}")

        # 입력 디바이스 선택 (우선순위: USB > 기본)
        usb_sources = [source for source in sources if source['is_usb']]
        if usb_sources:
            best_input = usb_sources[0]['name']
            logger.info(f"USB 입력 디바이스 선택: {best_input}")
        elif sources:
            # 모니터가 아닌 실제 입력 디바이스 선택
            real_sources = [source for source in sources if 'monitor' not in source['name'].lower()]
            if real_sources:
                best_input = real_sources[0]['name']
                logger.info(f"기본 입력 디바이스 선택: {best_input}")

        return best_output, best_input

    def monitor_device_changes(self, callback=None, interval: float = 5.0):
        """디바이스 변경 모니터링"""
        logger.info("디바이스 변경 모니터링 시작")

        previous_devices = set()

        try:
            while True:
                current_devices = set(self.scan_usb_devices().keys())

                # 새로 추가된 디바이스
                added = current_devices - previous_devices
                # 제거된 디바이스
                removed = previous_devices - current_devices

                if added or removed:
                    logger.info(f"디바이스 변경 감지 - 추가: {len(added)}, 제거: {len(removed)}")

                    if callback:
                        callback(added, removed)

                previous_devices = current_devices
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("디바이스 모니터링 중단")

    def get_device_summary(self) -> dict:
        """현재 디바이스 상태 요약"""
        stable_devices = self.get_stable_audio_devices(scan_count=1)
        sinks, sources = self.get_pulseaudio_devices()

        return {
            'usb_audio_devices': len([d for d in stable_devices.values() if d['is_audio']]),
            'usb_speakers': len([d for d in stable_devices.values() if d['device_type'] == 'speaker']),
            'usb_microphones': len([d for d in stable_devices.values() if d['device_type'] == 'microphone']),
            'pulse_sinks': len(sinks),
            'pulse_sources': len(sources),
            'usb_pulse_sinks': len([s for s in sinks if s['is_usb']]),
            'usb_pulse_sources': len([s for s in sources if s['is_usb']])
        }