"""실제 마이크·스피커를 사용해 음성을 녹음하고 재생하는 수동 테스트 스크립트."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

features_pkg = sys.modules.setdefault("features", types.ModuleType("features"))
features_pkg.__path__ = [str(PROJECT_ROOT / "features")]

qa_pkg = sys.modules.setdefault("features.qa", types.ModuleType("features.qa"))
qa_pkg.__path__ = [str(PROJECT_ROOT / "features" / "qa")]

managers_pkg = sys.modules.setdefault(
    "features.qa.managers", types.ModuleType("features.qa.managers")
)
managers_pkg.__path__ = [str(PROJECT_ROOT / "features" / "qa" / "managers")]

spec = importlib.util.spec_from_file_location(
    "features.qa.managers.voice_interface_manager",
    PROJECT_ROOT / "features" / "qa" / "managers" / "voice_interface_manager.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module  # type: ignore[arg-type]
spec.loader.exec_module(module)  # type: ignore[arg-type]

VoiceInterfaceManager = module.VoiceInterfaceManager


def main() -> None:
    manager = VoiceInterfaceManager()
    if not manager.is_initialized:
        print("❌ 음성 인터페이스 초기화 실패 (마이크/TTS 구성 확인 필요)")
        return

    if not manager.is_ready():
        print("⚠️ TTS 엔진이 비활성화되었습니다. 녹음만 진행합니다.")

    print("\n🎙 3초 동안 말을 녹음합니다. Enter 키를 누르면 시작합니다.")
    if sys.stdin.isatty():
        input("준비되면 Enter > ")
    else:
        print("⌛️ 비대화형 환경이므로 1초 후 자동으로 시작합니다.")
        time.sleep(1)

    audio_bytes = manager.listen_and_record()
    if not audio_bytes:
        print("❌ 음성이 감지되지 않았습니다. 다시 시도해 주세요.")
        return

    temp_dir = Path(tempfile.gettempdir())
    wav_path = temp_dir / "voice_roundtrip.wav"
    manager.mic_manager.save_audio_to_wav(audio_bytes, str(wav_path))
    print(f"✅ 녹음 완료: {wav_path}")

    print("📢 녹음한 음성을 재생합니다.")
    from features.qa.utils.audio import play_audio_file

    play_audio_file(str(wav_path))

    print("\n🗣  TTS 테스트를 진행합니다.")
    if manager.tts_manager.is_available():
        manager.speak("안녕하세요. 음성 인터페이스가 정상적으로 동작합니다.")
    else:
        print("⚠️ 사용 가능한 TTS 엔진이 없어 TTS 테스트를 건너뜁니다.")

    manager.cleanup()
    print("✅ 테스트 완료")


if __name__ == "__main__":
    main()
