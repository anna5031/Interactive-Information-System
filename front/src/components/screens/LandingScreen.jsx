import { useAppState } from '../../state/AppStateContext';
import layout from '../../styles/ScreenLayout.module.css';

const statusLabelMap = {
  connecting: '연결 중',
  connected: '연결됨',
  disconnected: '연결 끊김',
};

function LandingScreen() {
  const { connectionStatus } = useAppState();
  const statusLabel = statusLabelMap[connectionStatus] ?? connectionStatus;

  return (
    <div className={layout.screen}>
      <div className={layout.content}>
        <h1 className={layout.title}>안내 시스템 준비 중</h1>
        <p className={layout.status}>현재 상태: {statusLabel}</p>
        <p className={layout.hint}>연결이 완료되면 안내가 자동으로 시작됩니다.</p>
      </div>
    </div>
  );
}

export default LandingScreen;
