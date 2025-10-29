import { useMemo } from 'react';
import { useAppState } from '../../state/AppStateContext';
import layout from '../../styles/ScreenLayout.module.css';
import styles from './NudgeScreen.module.css';
import NudgeArrows from '../nudge/NudgeArrows';
import { homographyToCssMatrix, IDENTITY_MATRIX3D } from '../../utils/homography';

const detectionLabelMap = {
  assistance: '도움이 필요한 사람을 안내하는 중입니다.',
  guidance: '목적지 안내를 시작합니다.',
  scanning: '도움이 필요한 사람을 찾는 중입니다.',
  idle: '시스템이 대기 상태입니다.',
};

function NudgeScreen() {
  const { latestHomography, currentScreenCommand, detectionState } = useAppState();

  const displayPayload = useMemo(() => {
    if (!latestHomography) {
      return 'Homography 데이터를 기다리는 중입니다...';
    }
    return JSON.stringify(latestHomography.raw ?? latestHomography, null, 2);
  }, [latestHomography]);

  const transformValue = useMemo(() => {
    if (!latestHomography?.matrix) {
      return IDENTITY_MATRIX3D;
    }
    return homographyToCssMatrix(latestHomography.matrix) ?? IDENTITY_MATRIX3D;
  }, [latestHomography]);

  const overlayHeading = useMemo(() => {
    if (!detectionState) {
      return '안내 준비 중';
    }
    return detectionLabelMap[detectionState.status] ?? '안내 준비 중';
  }, [detectionState]);

  const showArrows = detectionState?.status === 'assistance' || detectionState?.status === 'guidance' || detectionState?.status === 'scanning';

  return (
    <div className={layout.screen}>
      <div className={`${layout.content} ${layout.contentWide}`}>
        <h1 className={layout.title}>프로젝터 안내 모드</h1>
        {currentScreenCommand?.context?.target && (
          <p className={layout.sub}>목적지: {currentScreenCommand.context.target}</p>
        )}

        <div className={styles.layout}>
          <div className={styles.preview}>
            <div className={styles.frame}>
              <div className={styles.canvas} style={{ transform: transformValue }}>
                <div className={styles.overlay}>
                  {showArrows && <NudgeArrows />}
                  <div className={styles.label}>
                    <div className={styles.infoIcon} aria-hidden="true">
                      <span>i</span>
                    </div>
                    <div className={styles.infoText}>
                      <span className={styles.infoHeading}>{overlayHeading}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <p className={styles.caption}>키스톤 변환 미리보기</p>
          </div>
          <div className={styles.dataWrapper}>
            <pre className={`${layout.data} ${styles.payload}`}>{displayPayload}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}

export default NudgeScreen;
