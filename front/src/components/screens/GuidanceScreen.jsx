import { useAppState } from '../../state/AppStateContext';
import layout from '../../styles/ScreenLayout.module.css';

function GuidanceScreen() {
  const { currentScreenCommand } = useAppState();
  const step = currentScreenCommand?.context?.step ?? null;
  const message = currentScreenCommand?.context?.message ?? '음성 안내를 재생 중입니다.';

  return (
    <div className={layout.screen}>
      <div className={layout.content}>
        <h1 className={layout.title}>안내 모드</h1>
        {step && <p className={layout.sub}>현재 단계: {step}</p>}
        <p className={layout.message}>{message}</p>
      </div>
    </div>
  );
}

export default GuidanceScreen;
