import { useMemo } from 'react';
import { useAppState } from '../../state/AppStateContext';
import layout from '../../styles/ScreenLayout.module.css';
import styles from './LandingScreen.module.css';
const detectionLabelMap = {
  assistance: '도움이 필요한 사람을 안내하는 중입니다.',
  guidance: '목적지 안내를 시작합니다.',
  scanning: '도움이 필요한 사람을 찾는 중입니다.',
  idle: '시스템이 대기 상태입니다.',
};

const statusLabelMap = {
  connecting: '연결 중',
  connected: '연결됨',
  disconnected: '연결 끊김',
};

function LandingScreen() {
  const { connectionStatus, detectionState } = useAppState();
  const statusLabel = statusLabelMap[connectionStatus] ?? connectionStatus;

  const detectionMessage = useMemo(() => {
    if (!detectionState) {
      return null;
    }
    return detectionLabelMap[detectionState.status] ?? null;
  }, [detectionState]);

  return (
    <div className={layout.screen}>
      <div className={layout.content}>
        <h1 className={layout.title}>안내 시스템 준비 중</h1>
        <p className={layout.status}>현재 상태: {statusLabel}</p>
        <p className={layout.hint}>연결이 완료되면 안내가 자동으로 시작됩니다.</p>

        {detectionMessage && <p className={styles.statusMessage}>{detectionMessage}</p>}
      </div>
    </div>
  );
}

export default LandingScreen;
