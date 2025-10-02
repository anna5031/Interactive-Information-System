import PropTypes from 'prop-types';
import { ArrowLeft, Clipboard, Download } from 'lucide-react';
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

const AnnotationReviewPage = ({ savedYoloText, savedWallText, onBack, onFinish }) => {
  const handleCopyAll = () => {
    const combined = [savedYoloText, savedWallText].filter(Boolean).join('\n\n');
    copyText(combined, '전체 결과');
  };

  const handleCopyYolo = () => copyText(savedYoloText, 'YOLO 박스 결과');
  const handleCopyWall = () => copyText(savedWallText, '벽 선 결과');

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <button type='button' className={styles.secondaryButton} onClick={onBack}>
          <ArrowLeft size={18} />
          수정으로 돌아가기
        </button>
        <div className={styles.titleGroup}>
          <h2 className={styles.title}>결과 확인</h2>
          <p className={styles.subtitle}>YOLO 박스와 벽(선) 데이터를 각각 내려받을 수 있습니다.</p>
        </div>
        <button type='button' className={styles.primaryButton} onClick={handleCopyAll}>
          <Clipboard size={18} />
          전체 복사
        </button>
      </header>

      <main className={styles.main}>
        <section className={styles.panel}>
          <div className={styles.panelHeader}>
            <h3 className={styles.panelTitle}>YOLO 박스 (yolo.txt)</h3>
            <button type='button' className={styles.copyButton} onClick={handleCopyYolo}>
              <Clipboard size={16} /> 복사
            </button>
          </div>
          <textarea className={styles.textarea} value={savedYoloText} readOnly />
        </section>

        <section className={styles.panel}>
          <div className={styles.panelHeader}>
            <h3 className={styles.panelTitle}>벽 선 (wall.txt)</h3>
            <button type='button' className={styles.copyButton} onClick={handleCopyWall}>
              <Clipboard size={16} /> 복사
            </button>
          </div>
          <textarea className={styles.textarea} value={savedWallText} readOnly />
        </section>
      </main>

      <footer className={styles.footer}>
        <button type='button' className={styles.finishButton} onClick={onFinish}>
          <Download size={18} />
          완료
        </button>
      </footer>
    </div>
  );
};

AnnotationReviewPage.propTypes = {
  savedYoloText: PropTypes.string,
  savedWallText: PropTypes.string,
  onBack: PropTypes.func.isRequired,
  onFinish: PropTypes.func.isRequired,
};

AnnotationReviewPage.defaultProps = {
  savedYoloText: '',
  savedWallText: '',
};

export default AnnotationReviewPage;
