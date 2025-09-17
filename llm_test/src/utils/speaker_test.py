#!/usr/bin/env python3
"""
USB 스피커 강제 설정 및 테스트
"""

import os
import subprocess
import tempfile
from gtts import gTTS
import time


class USBSpeakerFix:
    def __init__(self):
        self.usb_sink_name = None
        self.original_sink = None

    def get_pulseaudio_sinks(self):
        """PulseAudio 출력 디바이스 목록 가져오기"""
        try:
            result = subprocess.run(['pactl', 'list', 'short', 'sinks'],
                                    capture_output=True, text=True)
            return result.stdout.strip().split('\n')
        except:
            return []

    def get_current_default_sink(self):
        """현재 기본 출력 디바이스 확인"""
        try:
            result = subprocess.run(['pactl', 'info'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'Default Sink:' in line:
                    return line.split('Default Sink:')[1].strip()
        except:
            pass
        return None

    def find_usb_sink(self):
        """USB 스피커 sink 찾기"""
        sinks = self.get_pulseaudio_sinks()
        print("🔍 사용 가능한 PulseAudio 출력 디바이스:")

        usb_sinks = []
        for sink in sinks:
            if sink.strip():
                print(f"  {sink}")
                # USB 관련 키워드 검색
                if any(keyword in sink.lower() for keyword in ['usb', 'uac', 'demo', 'jieli']):
                    sink_name = sink.split('\t')[1]  # 두 번째 컬럼이 sink 이름
                    usb_sinks.append(sink_name)
                    print(f"    ⭐ USB 디바이스 발견: {sink_name}")

        return usb_sinks

    def set_usb_as_default(self, usb_sink):
        """USB 스피커를 기본 출력으로 설정"""
        try:
            # 현재 기본 디바이스 저장
            self.original_sink = self.get_current_default_sink()
            print(f"📝 현재 기본 출력: {self.original_sink}")

            # USB 스피커를 기본으로 설정
            result = subprocess.run(['pactl', 'set-default-sink', usb_sink],
                                    capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ USB 스피커를 기본 출력으로 설정: {usb_sink}")
                self.usb_sink_name = usb_sink

                # 설정 확인
                time.sleep(1)
                current_sink = self.get_current_default_sink()
                print(f"🔄 변경된 기본 출력: {current_sink}")
                return True
            else:
                print(f"❌ USB 스피커 설정 실패: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ USB 스피커 설정 중 오류: {e}")
            return False

    def test_audio_output(self, test_text="USB 스피커 테스트입니다. 이 소리가 들리나요?"):
        """다양한 방법으로 오디오 출력 테스트"""
        print(f"\n🎵 오디오 출력 테스트 시작...")

        # TTS 파일 생성
        tts = gTTS(text=test_text, lang='ko')
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            mp3_file = temp_file.name
            tts.save(mp3_file)

            # WAV 변환
            wav_file = mp3_file.replace('.mp3', '.wav')
            os.system(f"ffmpeg -i '{mp3_file}' -acodec pcm_s16le -ar 48000 '{wav_file}' -y 2>/dev/null")

            success_methods = []

            # 테스트 방법들
            test_methods = [
                ("paplay 기본", f"paplay '{mp3_file}'"),
                ("paplay WAV", f"paplay '{wav_file}'"),
                ("paplay USB 지정", f"paplay --device={self.usb_sink_name} '{mp3_file}'" if self.usb_sink_name else None),
                ("mpg123 기본", f"mpg123 -q '{mp3_file}'"),
                ("mpg123 pulse", f"mpg123 -a pulse -q '{mp3_file}'"),
                ("aplay Card 0", f"aplay -D hw:0,0 '{wav_file}'"),
                ("aplay plughw:0", f"aplay -D plughw:0,0 '{wav_file}'"),
                ("speaker-test", "speaker-test -t sine -f 1000 -l 1 -D hw:0,0"),
            ]

            for method_name, command in test_methods:
                if command is None:  # USB sink가 없는 경우 skip
                    continue

                print(f"\n🔊 {method_name} 테스트...")
                try:
                    if "speaker-test" in command:
                        # speaker-test는 별도 처리 (sine wave 테스트)
                        print("  📢 1초간 1000Hz 사인파 재생...")
                        result = os.system(f"timeout 1s {command} 2>/dev/null")
                    else:
                        result = os.system(f"{command} 2>/dev/null")

                    if result == 0:
                        print(f"  ✅ {method_name} 성공!")
                        success_methods.append((method_name, command))
                    else:
                        print(f"  ❌ {method_name} 실패")

                except Exception as e:
                    print(f"  ❌ {method_name} 오류: {e}")

            # 임시 파일 정리
            for file_path in [mp3_file, wav_file]:
                if os.path.exists(file_path):
                    os.unlink(file_path)

            return success_methods

    def restore_original_sink(self):
        """원래 기본 출력으로 복원"""
        if self.original_sink:
            try:
                subprocess.run(['pactl', 'set-default-sink', self.original_sink])
                print(f"🔄 기본 출력을 원래대로 복원: {self.original_sink}")
            except:
                pass

    def run_complete_test(self):
        """전체 USB 스피커 설정 및 테스트"""
        print("🎯 USB 스피커 완전 설정 및 테스트 시작\n")
        print("=" * 60)

        # 1. USB sink 찾기
        usb_sinks = self.find_usb_sink()

        if not usb_sinks:
            print("❌ USB 스피커를 찾을 수 없습니다.")
            return

        # 2. 첫 번째 USB sink를 기본으로 설정
        usb_sink = usb_sinks[0]
        print(f"\n🔧 USB 스피커 설정 중: {usb_sink}")

        if not self.set_usb_as_default(usb_sink):
            print("❌ USB 스피커 설정에 실패했습니다.")
            return

        # 3. 오디오 출력 테스트
        print("\n" + "=" * 60)
        success_methods = self.test_audio_output()

        # 4. 결과 요약
        print("\n" + "=" * 60)
        print("🏁 테스트 완료!")

        if success_methods:
            print(f"✅ {len(success_methods)}개 방법으로 재생 성공!")
            print("💡 성공한 방법들:")
            for method_name, command in success_methods:
                print(f"   - {method_name}")

            # 가장 좋은 방법 추천
            preferred_order = ['paplay USB 지정', 'paplay 기본', 'mpg123 기본', 'aplay Card 0']
            best_method = None

            for preferred in preferred_order:
                for method_name, command in success_methods:
                    if preferred in method_name:
                        best_method = (method_name, command)
                        break
                if best_method:
                    break

            if not best_method:
                best_method = success_methods[0]

            print(f"\n🎯 추천 방법: {best_method[0]}")
            print(f"📋 명령어: {best_method[1]}")

            return best_method

        else:
            print("❌ 모든 재생 방법이 실패했습니다.")
            print("💡 문제 해결 시도:")
            print("   1. USB 스피커가 제대로 연결되어 있는지 확인")
            print("   2. 볼륨이 음소거되어 있지 않은지 확인")
            print("   3. 다른 애플리케이션이 오디오를 사용하고 있는지 확인")

        return None


def main():
    fixer = USBSpeakerFix()

    try:
        best_method = fixer.run_complete_test()

        if best_method:
            print(f"\n🚀 이 설정을 메인 음성 AI에 적용하세요!")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n👋 사용자에 의해 중단되었습니다.")
    finally:
        # 원래 설정으로 복원할지 물어보기
        try:
            restore = input("\n❓ 기본 오디오 출력을 원래대로 복원하시겠습니까? (y/N): ").lower()
            if restore == 'y':
                fixer.restore_original_sink()
            else:
                print("💡 USB 스피커가 기본 출력으로 설정된 상태로 유지됩니다.")
        except:
            pass


if __name__ == "__main__":
    main()