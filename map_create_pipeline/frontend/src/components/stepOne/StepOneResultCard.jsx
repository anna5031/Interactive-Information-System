import PropTypes from 'prop-types';
import { ClipboardList, GitBranch, Image as ImageIcon, Loader2, NotebookPen, PenLine, Trash2 } from 'lucide-react';
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

const StepOneResultCard = ({
  result,
  onSelectStepThree,
  onEditStepOne,
  onOpenGraph,
  stepThreeStatus,
  isGraphOpening,
  onDelete,
  isDeleting,
}) => {
  const imageUrl = result?.metadata?.imageUrl || result?.imageUrl || result?.imageDataUrl || null;
  const floorLabel = (result?.metadata?.floorLabel || result?.floorLabel || '').trim();
  const fallbackTitle = result?.metadata?.fileName || result?.fileName || result?.id;
  const title = floorLabel || fallbackTitle;
  const createdAt = formatTimestamp(result?.createdAt);
  const boxesCount = Array.isArray(result?.objectDetection?.boxes) ? result.objectDetection.boxes.length : 0;
  const linesCount = Array.isArray(result?.wall?.lines) ? result.wall.lines.length : 0;
  const pointsCount = Array.isArray(result?.door?.points) ? result.door.points.length : 0;
  const processingRequestId = result?.processingResult?.request_id ?? result?.requestId ?? null;
  const hasValidGraph = Boolean(result?.processingResult?.metadata?.graph_summary);
  const graphStatusLabel = hasValidGraph ? '생성 완료' : '미생성 (2단계에서 생성)';
  const hasStepThreeBase = Boolean(stepThreeStatus?.hasBase);
  const hasStepThreeDetails = Boolean(stepThreeStatus?.hasDetails);
  const baseStatusLabel = hasStepThreeBase ? '입력 완료' : hasValidGraph ? '입력 필요' : '그래프 필요';
  const canEditGraph = !isGraphOpening;
  const canEditBase = hasValidGraph;
  const canEditDetails = hasStepThreeBase && hasValidGraph;
  const canDelete = Boolean(processingRequestId);
  const baseButtonLabel = hasStepThreeBase ? '(3단계) 기본 정보 수정' : '(3단계) 기본 정보 입력';
  const detailButtonLabel = hasStepThreeDetails ? '(3단계) 상세 정보 수정' : '(3단계) 상세 정보 입력';

  const handleSelectStepThree = (targetStage = 'base') => {
    if (!hasValidGraph) {
      // eslint-disable-next-line no-alert
      alert('그래프가 생성되지 않았습니다. 2단계(그래프 편집)에서 그래프를 저장해 주세요.');
      return;
    }
    const needsBaseFirst = targetStage === 'details' && !hasStepThreeBase;
    const resolvedStage = needsBaseFirst ? 'base' : targetStage;
    if (needsBaseFirst) {
      // eslint-disable-next-line no-alert
      alert('상세 정보를 입력하려면 먼저 기본 정보를 완료해 주세요.');
    }
    onSelectStepThree?.(result, resolvedStage);
  };

  const handleEditClick = () => {
    const hasStepThreeProgress = Boolean(stepThreeStatus?.hasBase || stepThreeStatus?.hasDetails);
    if (hasStepThreeProgress) {
      // eslint-disable-next-line no-alert
      const confirmed = window.confirm(
        '1단계를 수정하면 변경 내용에 따라 2, 3단계 진행 내용이 초기화되거나 변경될 수 있습니다. 계속하시겠습니까?'
      );
      if (!confirmed) {
        return;
      }
    }
    onEditStepOne?.(result);
  };

  const handleDeleteClick = () => {
    if (!canDelete) {
      // eslint-disable-next-line no-alert
      alert('삭제할 요청 정보를 찾지 못했습니다.');
      return;
    }
    // eslint-disable-next-line no-alert
    const confirmed = window.confirm('삭제한 도면 카드는 복구할 수 없습니다. 정말 삭제하시겠습니까?');
    if (!confirmed) {
      return;
    }
    onDelete?.(result);
  };

  return (
    <article className={styles.card}>
      <button
        type='button'
        className={styles.deleteIconButton}
        onClick={handleDeleteClick}
        disabled={!canDelete || isDeleting}
        title={!canDelete ? '삭제할 수 있는 요청 정보가 없습니다.' : '카드 삭제'}
        aria-label='카드 삭제'
      >
        {isDeleting ? <Loader2 className={styles.deleteSpinner} size={18} /> : <Trash2 size={20} />}
      </button>
      <div className={styles.thumbnail}>
        {imageUrl ? <img src={imageUrl} alt={`${title} preview`} /> : <ImageIcon size={32} />}
      </div>
      <div className={styles.content}>
        <h3 className={styles.title}>
          {title}
          {processingRequestId ? <span className={styles.requestId}>#{processingRequestId}</span> : null}
        </h3>
        <p className={styles.subtitle}>{createdAt}</p>
        {floorLabel && fallbackTitle ? <p className={styles.fileInfo}>파일명 {fallbackTitle}</p> : null}
        <div className={styles.metrics}>
          <span>박스 {boxesCount}</span>
          <span>선 {linesCount}</span>
          <span>포인트 {pointsCount}</span>
        </div>
        <div
          className={`${styles.processingStatus} ${
            hasValidGraph ? styles.processingStatusReady : styles.processingStatusMissing
          }`}
        >
          <strong>그래프</strong>
          <span>{graphStatusLabel}</span>
        </div>
        <div className={styles.stageStatusList}>
          <div
            className={`${styles.stageStatusItem} ${
              hasStepThreeBase ? styles.stageStatusReady : styles.stageStatusPending
            }`}
          >
            <strong>기본 정보</strong>
            <span>{baseStatusLabel}</span>
          </div>
        </div>
      </div>
      <div className={styles.actions}>
        <button
          type='button'
          className={`${styles.actionButton} ${styles.editButton}`}
          onClick={handleEditClick}
        >
          <ClipboardList size={16} /> (1단계) 라벨링 수정
        </button>
        <button
          type='button'
          className={`${styles.actionButton} ${styles.graphButton}`}
          onClick={() => onOpenGraph?.(result)}
          disabled={!canEditGraph}
        >
          <GitBranch size={16} /> {isGraphOpening ? '그래프 준비중...' : '(2단계) 그래프 편집'}
        </button>
        <button
          type='button'
          className={`${styles.actionButton} ${styles.baseButton}`}
          onClick={() => handleSelectStepThree('base')}
          disabled={!canEditBase}
        >
          <PenLine size={16} /> {baseButtonLabel}
        </button>
        <button
          type='button'
          className={`${styles.actionButton} ${styles.detailButton}`}
          onClick={() => handleSelectStepThree('details')}
          disabled={!canEditDetails}
          title={
            !hasStepThreeBase
              ? '기본 정보를 완료한 뒤 상세 정보를 입력할 수 있습니다.'
              : hasStepThreeDetails
                ? '저장된 상세 정보를 수정합니다.'
                : '상세 정보를 새로 입력할 수 있습니다.'
          }
        >
          <NotebookPen size={16} /> {detailButtonLabel}
        </button>
      </div>
    </article>
  );
};

StepOneResultCard.propTypes = {
  result: PropTypes.shape({
    id: PropTypes.string.isRequired,
    fileName: PropTypes.string,
    floorLabel: PropTypes.string,
    floorValue: PropTypes.string,
    createdAt: PropTypes.string,
    imageUrl: PropTypes.string,
    imageDataUrl: PropTypes.string,
    metadata: PropTypes.shape({
      fileName: PropTypes.string,
      imageUrl: PropTypes.string,
      floorLabel: PropTypes.string,
      floorValue: PropTypes.string,
    }),
    processingResult: PropTypes.object,
    requestId: PropTypes.string,
    objectDetection: PropTypes.shape({ boxes: PropTypes.array }),
    wall: PropTypes.shape({ lines: PropTypes.array }),
    door: PropTypes.shape({ points: PropTypes.array }),
  }).isRequired,
  onSelectStepThree: PropTypes.func,
  onOpenGraph: PropTypes.func,
  onEditStepOne: PropTypes.func,
  stepThreeStatus: PropTypes.shape({
    hasBase: PropTypes.bool,
    hasDetails: PropTypes.bool,
    baseUpdatedAt: PropTypes.string,
    detailsUpdatedAt: PropTypes.string,
  }),
  isGraphOpening: PropTypes.bool,
  onDelete: PropTypes.func,
  isDeleting: PropTypes.bool,
};

StepOneResultCard.defaultProps = {
  onSelectStepThree: () => {},
  onOpenGraph: () => {},
  onEditStepOne: () => {},
  stepThreeStatus: null,
  isGraphOpening: false,
  onDelete: () => {},
  isDeleting: false,
};

export default StepOneResultCard;
