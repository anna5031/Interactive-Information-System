"""TTS 엔진 팩토리 - 모듈식 TTS 엔진 선택 및 관리"""

from typing import Dict, List, Optional

from .gtts_engine import GTTSEngine
from .elevenlabs_engine import ElevenLabsEngine


class TTSFactory:
    """TTS 엔진 팩토리 클래스"""

    def __init__(self):
        self._engine_classes = {
            "gtts": GTTSEngine,
            "elevenlabs": ElevenLabsEngine,
        }
        self._engines = {}
        self._load_engines()

    def _load_engines(self):
        print("🔧 TTS 엔진 로드 중...")

        for engine_id, engine_class in self._engine_classes.items():
            if engine_class is None:
                continue

            try:
                print(f"   {engine_id} 엔진 초기화...")
                engine = engine_class()
                self._engines[engine_id] = engine

                if engine.is_available():
                    print(f"   ✅ {engine.name} 사용 가능")
                else:
                    print(f"   ❌ {engine.name} 사용 불가")

            except Exception as exc:
                print(f"   ❌ {engine_id} 엔진 로드 실패: {exc}")

    def get_engine(self, engine_id: str):
        return self._engines.get(engine_id)

    def get_available_engines(self) -> List[Dict]:
        available = []
        for engine_id, engine in self._engines.items():
            if engine.is_available():
                info = engine.get_info()
                info["engine_id"] = engine_id
                available.append(info)
        return available

    def get_all_engines(self) -> List[Dict]:
        all_engines = []
        for engine_id, engine in self._engines.items():
            info = engine.get_info()
            info["engine_id"] = engine_id
            all_engines.append(info)
        return all_engines

    def synthesize(
        self,
        text: str,
        engine_id: str,
        language: str,
        output_file: Optional[str] = None,
    ) -> bool:
        engine = self.get_engine(engine_id)
        if not engine:
            print(f"❌ 엔진을 찾을 수 없습니다: {engine_id}")
            return False

        if not engine.is_available():
            print(f"❌ {engine.name} 엔진을 사용할 수 없습니다.")
            return False

        return engine.synthesize(text, language, output_file)

    def synthesize_and_play(self, text: str, engine_id: str, language: str) -> bool:
        engine = self.get_engine(engine_id)
        if not engine:
            print(f"❌ 엔진을 찾을 수 없습니다: {engine_id}")
            return False

        if not engine.is_available():
            print(f"❌ {engine.name} 엔진을 사용할 수 없습니다.")
            return False

        return engine.synthesize_and_play(text, language)

    def test_engine(self, engine_id: str, test_text: Optional[str] = None) -> bool:
        engine = self.get_engine(engine_id)
        if not engine:
            print(f"❌ 엔진을 찾을 수 없습니다: {engine_id}")
            return False

        if not engine.is_available():
            print(f"❌ {engine.name} 엔진을 사용할 수 없습니다.")
            return False

        return engine.test(test_text or "안녕하세요. TTS 테스트입니다.")

    def test_all_engines(self) -> Dict[str, bool]:
        results = {}
        available_engines = self.get_available_engines()

        if not available_engines:
            print("❌ 사용 가능한 엔진이 없습니다.")
            return results

        print("🧪 모든 엔진 테스트 시작...")

        for engine_info in available_engines:
            engine_id = engine_info["engine_id"]
            print(f"\n📢 {engine_info['name']} 테스트:")
            results[engine_id] = self.test_engine(engine_id)

        return results

    def select_best_engine(self) -> Optional[str]:
        available = self.get_available_engines()

        if not available:
            return None

        for engine_info in available:
            if engine_info["engine_id"] == "elevenlabs":
                return "elevenlabs"

        for engine_info in available:
            if engine_info["engine_id"] == "gtts":
                return "gtts"

        return available[0]["engine_id"]

    def interactive_selection(self) -> Optional[str]:
        available = self.get_available_engines()

        if not available:
            print("❌ 사용 가능한 엔진이 없습니다.")
            return None

        if len(available) == 1:
            engine_info = available[0]
            print(f"🎯 유일한 사용 가능 엔진: {engine_info['name']}")
            return engine_info["engine_id"]

        print("\n🎤 사용 가능한 TTS 엔진:")
        for i, engine_info in enumerate(available):
            print(f"{i + 1}. {engine_info['name']} ({engine_info['description']})")

        while True:
            try:
                choice = input("\n엔진 선택 (번호): ").strip()
                index = int(choice) - 1

                if 0 <= index < len(available):
                    selected = available[index]
                    print(f"✅ {selected['name']} 선택됨")
                    return selected["engine_id"]
                else:
                    print("❌ 잘못된 번호입니다.")

            except ValueError:
                print("❌ 숫자를 입력해주세요.")
            except KeyboardInterrupt:
                print("\n❌ 취소됨")
                return None

    def get_status(self) -> Dict:
        available_engines = self.get_available_engines()
        all_engines = self.get_all_engines()

        return {
            "total_engines": len(all_engines),
            "available_engines": len(available_engines),
            "engines": all_engines,
            "best_engine": self.select_best_engine(),
        }
