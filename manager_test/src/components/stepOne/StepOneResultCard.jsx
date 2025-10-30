import PropTypes from 'prop-types';
import { ClipboardList, Download, Image as ImageIcon, PenLine } from 'lucide-react';
import { downloadJson } from '../../utils/download';
import styles from './StepOneResultCard.module.css';

const formatTimestamp = (value) => {
  if (!value) {
    return '-';
  }
  try {
    return new Date(value).toLocaleString('ko-KR');
  } catch (error) {
    return value;
  }
};

const StepOneResultCard = ({ result, onSelectStepTwo, onEditStepOne }) => {
  const imageUrl = result?.metadata?.imageUrl;
  const title = result?.metadata?.fileName || result?.fileName || result?.id;
  const createdAt = formatTimestamp(result?.createdAt);
  const boxesCount = Array.isArray(result?.yolo?.boxes) ? result.yolo.boxes.length : 0;
  const linesCount = Array.isArray(result?.wall?.lines) ? result.wall.lines.length : 0;
  const pointsCount = Array.isArray(result?.door?.points) ? result.door.points.length : 0;

  const handleDownload = () => {
    downloadJson(result.fileName ?? result.id ?? 'step_one_result', result);
  };

  return (
    <article className={styles.card}>
      <div className={styles.thumbnail}>
        {imageUrl ? <img src={imageUrl} alt={`${title} preview`} /> : <ImageIcon size={32} />}
      </div>
      <div className={styles.content}>
        <h3 className={styles.title}>{title}</h3>
        <p className={styles.subtitle}>{createdAt}</p>
        <div className={styles.metrics}>
          <span>박스 {boxesCount}</span>
          <span>선 {linesCount}</span>
          <span>포인트 {pointsCount}</span>
        </div>
      </div>
      <div className={styles.actions}>
        <button type='button' className={styles.mutedButton} onClick={() => onEditStepOne?.(result)}>
          <ClipboardList size={16} /> 1단계 수정
        </button>
        <button type='button' className={styles.primaryButton} onClick={() => onSelectStepTwo?.(result)}>
          <PenLine size={16} /> 2단계 진행
        </button>
        <button type='button' className={styles.secondaryButton} onClick={handleDownload}>
          <Download size={16} /> JSON 다운로드
        </button>
      </div>
    </article>
  );
};

StepOneResultCard.propTypes = {
  result: PropTypes.shape({
    id: PropTypes.string.isRequired,
    fileName: PropTypes.string,
    createdAt: PropTypes.string,
    metadata: PropTypes.shape({
      fileName: PropTypes.string,
      imageUrl: PropTypes.string,
    }),
    yolo: PropTypes.shape({ boxes: PropTypes.array }),
    wall: PropTypes.shape({ lines: PropTypes.array }),
    door: PropTypes.shape({ points: PropTypes.array }),
  }).isRequired,
  onSelectStepTwo: PropTypes.func,
  onEditStepOne: PropTypes.func,
};

StepOneResultCard.defaultProps = {
  onSelectStepTwo: () => {},
  onEditStepOne: () => {},
};

export default StepOneResultCard;
