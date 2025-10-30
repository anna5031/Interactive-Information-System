import { useMemo } from 'react';
import { useAppState } from '../../state/AppStateContext';
import layout from '../../styles/ScreenLayout.module.css';
import styles from './NudgeScreen.module.css';
import NudgeArrows from '../nudge/NudgeArrows';
import { homographyToCssMatrix, IDENTITY_MATRIX3D } from '../../utils/homography';

const NORMALISED_CORNERS = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1],
];
const MAX_PROJECTED_DIMENSION = 1.35;

function projectPoint(matrix, x, y) {
  const [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]] = matrix;
  const w = h31 * x + h32 * y + h33;
  if (!Number.isFinite(w) || Math.abs(w) < 1e-6) {
    return null;
  }

  const px = (h11 * x + h12 * y + h13) / w;
  const py = (h21 * x + h22 * y + h23) / w;

  if (!Number.isFinite(px) || !Number.isFinite(py)) {
    return null;
  }
  return [px, py];
}

function computeContainScale(matrix) {
  if (!Array.isArray(matrix) || matrix.length !== 3) {
    return 1;
  }

  const projected = NORMALISED_CORNERS.map(([x, y]) => projectPoint(matrix, x, y)).filter(
    (point) => point !== null,
  );

  if (projected.length !== NORMALISED_CORNERS.length) {
    return 1;
  }

  const xs = projected.map(([px]) => px);
  const ys = projected.map(([, py]) => py);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);
  const maxDimension = Math.max(width, height);

  if (!Number.isFinite(maxDimension) || maxDimension <= 1) {
    return 1;
  }

  if (maxDimension <= MAX_PROJECTED_DIMENSION) {
    return 1;
  }

  return MAX_PROJECTED_DIMENSION / maxDimension;
}

function buildTransform(matrix) {
  if (!matrix) {
    return IDENTITY_MATRIX3D;
  }
  const cssMatrix = homographyToCssMatrix(matrix);
  if (!cssMatrix) {
    return IDENTITY_MATRIX3D;
  }

  const scale = computeContainScale(matrix);
  if (scale < 0.999) {
    return `${cssMatrix} scale(${scale})`;
  }
  return cssMatrix;
}

function NudgeScreen() {
  const { latestHomography } = useAppState();

  const stageTransform = useMemo(
    () => buildTransform(latestHomography?.matrix),
    [latestHomography?.matrix],
  );

  return (
    <div className={layout.screen}>
      <div className={`${layout.content} ${styles.content}`}>
        <div className={styles.stageWrapper}>
          <div className={styles.stageInner} style={{ transform: stageTransform }}>
            <div className={styles.stageOverlay}>
              <NudgeArrows />
              <div className={styles.label}>
                <div className={styles.infoIcon} aria-hidden="true">
                  <span>i</span>
                </div>
                <div className={styles.infoText}>
                  <span className={styles.infoHeading}>도움이 필요하신가요?</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default NudgeScreen;
