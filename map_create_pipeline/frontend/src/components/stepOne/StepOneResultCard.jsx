import PropTypes from 'prop-types';
import { ClipboardList, Download, FileText, Image as ImageIcon, NotebookPen, PenLine } from 'lucide-react';
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

const StepOneResultCard = ({ result, onSelectStepTwo, onEditStepOne, stepTwoStatus }) => {
  const imageUrl = result?.metadata?.imageUrl || result?.imageUrl || result?.imageDataUrl || null;
  const title = result?.metadata?.fileName || result?.fileName || result?.id;
  const createdAt = formatTimestamp(result?.createdAt);
  const boxesCount = Array.isArray(result?.yolo?.boxes) ? result.yolo.boxes.length : 0;
  const linesCount = Array.isArray(result?.wall?.lines) ? result.wall.lines.length : 0;
  const pointsCount = Array.isArray(result?.door?.points) ? result.door.points.length : 0;
  const processingRequestId = result?.processingResult?.request_id ?? null;
  const canProceedToStepTwo = Boolean(processingRequestId);
  const hasStepTwoBase = Boolean(stepTwoStatus?.hasBase);
  const hasStepTwoDetails = Boolean(stepTwoStatus?.hasDetails);

  const handleDownload = () => {
    downloadJson(result.fileName ?? result.id ?? 'step_one_result', result);
  };

  const handleSelectStepTwo = (targetStage = 'base') => {
    if (!canProceedToStepTwo) {
      return;
    }
    const needsBaseFirst = targetStage === 'details' && !hasStepTwoBase;
    const resolvedStage = needsBaseFirst ? 'base' : targetStage;
    if (needsBaseFirst) {
      // eslint-disable-next-line no-alert
      alert('상세 정보를 입력하려면 먼저 기본 정보를 완료해 주세요.');
    }
    onSelectStepTwo?.(result, resolvedStage);
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
        <div
          className={`${styles.processingStatus} ${
            canProceedToStepTwo ? styles.processingStatusReady : styles.processingStatusMissing
          }`}
        >
          <strong>그래프</strong>
          <span>{processingRequestId ?? '미생성'}</span>
        </div>
      </div>
      <div className={styles.actions}>
        <button type='button' className={styles.mutedButton} onClick={() => onEditStepOne?.(result)}>
          <ClipboardList size={16} /> 1단계 수정
        </button>
        <button
          type='button'
          className={styles.primaryButton}
          onClick={() => handleSelectStepTwo('base')}
          disabled={!canProceedToStepTwo}
        >
          <PenLine size={16} /> 2단계 진행
        </button>
        <button
          type='button'
          className={styles.mutedButton}
          onClick={() => handleSelectStepTwo('base')}
          disabled={!canProceedToStepTwo || !hasStepTwoBase}
        >
          <FileText size={16} /> 기본정보 수정
        </button>
        <button
          type='button'
          className={styles.mutedButton}
          onClick={() => handleSelectStepTwo('details')}
          disabled={!canProceedToStepTwo}
          title={
            !hasStepTwoBase
              ? '기본 정보를 완료한 뒤 상세 정보를 입력할 수 있습니다.'
              : hasStepTwoDetails
                ? '저장된 상세 정보를 수정합니다.'
                : '상세 정보를 새로 입력할 수 있습니다.'
          }
        >
          <NotebookPen size={16} /> 상세정보 수정
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
    imageUrl: PropTypes.string,
    imageDataUrl: PropTypes.string,
    metadata: PropTypes.shape({
      fileName: PropTypes.string,
      imageUrl: PropTypes.string,
    }),
    processingResult: PropTypes.object,
    yolo: PropTypes.shape({ boxes: PropTypes.array }),
    wall: PropTypes.shape({ lines: PropTypes.array }),
    door: PropTypes.shape({ points: PropTypes.array }),
  }).isRequired,
  onSelectStepTwo: PropTypes.func,
  onEditStepOne: PropTypes.func,
  stepTwoStatus: PropTypes.shape({
    hasBase: PropTypes.bool,
    hasDetails: PropTypes.bool,
    baseUpdatedAt: PropTypes.string,
    detailsUpdatedAt: PropTypes.string,
  }),
};

StepOneResultCard.defaultProps = {
  onSelectStepTwo: () => {},
  onEditStepOne: () => {},
  stepTwoStatus: null,
};

export default StepOneResultCard;
