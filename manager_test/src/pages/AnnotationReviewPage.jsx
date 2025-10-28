import PropTypes from 'prop-types';
import { ArrowLeft, Clipboard, FileJson2 } from 'lucide-react';
import { useMemo } from 'react';
import styles from './AnnotationReviewPage.module.css';

const copyText = async (text, message) => {
  if (!text) {
    // eslint-disable-next-line no-alert
    alert('복사할 내용이 없습니다.');
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    // eslint-disable-next-line no-alert
    alert(`${message}이(가) 복사되었습니다.`);
  } catch (error) {
    console.error('Failed to copy text', error);
    // eslint-disable-next-line no-alert
    alert('복사에 실패했습니다.');
  }
};

const formatJson = (value) => {
  try {
    return JSON.stringify(value ?? {}, null, 2);
  } catch (error) {
    console.error('Failed to stringify JSON', error);
    return '{}';
  }
};

const AnnotationReviewPage = ({ stepOneResult, onBack, onFinish }) => {
  const combinedPayload = useMemo(
    () => ({
      yolo: stepOneResult?.yolo ?? {},
      wall: stepOneResult?.wall ?? {},
      door: stepOneResult?.door ?? {},
    }),
    [stepOneResult]
  );

  const combinedJson = useMemo(() => formatJson(combinedPayload), [combinedPayload]);
  const yoloJson = useMemo(() => formatJson(stepOneResult?.yolo), [stepOneResult]);
  const wallJson = useMemo(() => formatJson(stepOneResult?.wall), [stepOneResult]);
  const doorJson = useMemo(() => formatJson(stepOneResult?.door), [stepOneResult]);

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <button type='button' className={styles.secondaryButton} onClick={onBack}>
          <ArrowLeft size={18} />
          수정으로 돌아가기
        </button>
        <div className={styles.titleGroup}>
          <h2 className={styles.title}>1단계 결과 확인</h2>
          <p className={styles.subtitle}>저장된 JSON을 확인하고 복사할 수 있습니다.</p>
        </div>
        <button type='button' className={styles.primaryButton} onClick={() => copyText(combinedJson, '통합 JSON')}>
          <Clipboard size={18} /> 통합 JSON 복사
        </button>
      </header>

      <main className={styles.main}>
        <section className={styles.panel}>
          <div className={styles.panelHeader}>
            <h3 className={styles.panelTitle}>
              <FileJson2 size={18} /> 저장 정보
            </h3>
          </div>
          <dl className={styles.metaList}>
            <div className={styles.metaRow}>
              <dt>파일명</dt>
              <dd>{stepOneResult?.fileName ?? '-'}</dd>
            </div>
            <div className={styles.metaRow}>
              <dt>경로(참고)</dt>
              <dd>{stepOneResult?.filePath ?? 'src/dummy/step_one_result/<uuid>.json'}</dd>
            </div>
            <div className={styles.metaRow}>
              <dt>저장 시각</dt>
              <dd>{stepOneResult?.createdAt ?? '-'}</dd>
            </div>
            <div className={styles.metaRow}>
              <dt>원본 도면</dt>
              <dd>{stepOneResult?.metadata?.fileName ?? '-'}</dd>
            </div>
          </dl>
        </section>

        <section className={styles.panel}>
          <div className={styles.panelHeader}>
            <h3 className={styles.panelTitle}>통합 JSON</h3>
            <button type='button' className={styles.copyButton} onClick={() => copyText(combinedJson, '통합 JSON')}>
              <Clipboard size={16} /> 복사
            </button>
          </div>
          <textarea className={styles.textarea} value={combinedJson} readOnly />
        </section>

        <section className={styles.panel}>
          <div className={styles.panelHeader}>
            <h3 className={styles.panelTitle}>세부 - YOLO</h3>
            <button type='button' className={styles.copyButton} onClick={() => copyText(yoloJson, 'YOLO 결과')}>
              <Clipboard size={16} /> 복사
            </button>
          </div>
          <textarea className={styles.textarea} value={yoloJson} readOnly />
        </section>

        <section className={styles.panel}>
          <div className={styles.panelHeader}>
            <h3 className={styles.panelTitle}>세부 - Wall</h3>
            <button type='button' className={styles.copyButton} onClick={() => copyText(wallJson, '벽 결과')}>
              <Clipboard size={16} /> 복사
            </button>
          </div>
          <textarea className={styles.textarea} value={wallJson} readOnly />
        </section>

        <section className={styles.panel}>
          <div className={styles.panelHeader}>
            <h3 className={styles.panelTitle}>세부 - Door</h3>
            <button type='button' className={styles.copyButton} onClick={() => copyText(doorJson, '문 결과')}>
              <Clipboard size={16} /> 복사
            </button>
          </div>
          <textarea className={styles.textarea} value={doorJson} readOnly />
        </section>
      </main>

      <footer className={styles.footer}>
        <button type='button' className={styles.finishButton} onClick={onFinish}>
          완료
        </button>
      </footer>
    </div>
  );
};

AnnotationReviewPage.propTypes = {
  stepOneResult: PropTypes.shape({
    id: PropTypes.string,
    fileName: PropTypes.string,
    filePath: PropTypes.string,
    createdAt: PropTypes.string,
    yolo: PropTypes.object,
    wall: PropTypes.object,
    door: PropTypes.object,
    metadata: PropTypes.shape({
      fileName: PropTypes.string,
      imageUrl: PropTypes.string,
    }),
  }),
  onBack: PropTypes.func.isRequired,
  onFinish: PropTypes.func.isRequired,
};

AnnotationReviewPage.defaultProps = {
  stepOneResult: null,
};

export default AnnotationReviewPage;
