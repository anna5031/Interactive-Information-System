import PropTypes from 'prop-types';
import { ArrowLeft, Clipboard, Download } from 'lucide-react';
import styles from './AnnotationReviewPage.module.css';

const AnnotationReviewPage = ({ savedText, onBack, onFinish }) => {
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(savedText);
      // eslint-disable-next-line no-alert
      alert('복사되었습니다.');
    } catch (error) {
      console.error('Failed to copy text', error);
      // eslint-disable-next-line no-alert
      alert('복사에 실패했습니다.');
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <button type='button' className={styles.secondaryButton} onClick={onBack}>
          <ArrowLeft size={18} />
          수정으로 돌아가기
        </button>
        <div className={styles.titleGroup}>
          <h2 className={styles.title}>YOLO 아웃풋</h2>
          <p className={styles.subtitle}>아래 내용을 복사하여 다른 툴에서 활용하세요.</p>
        </div>
        <button type='button' className={styles.primaryButton} onClick={handleCopy}>
          <Clipboard size={18} />
          전체 복사
        </button>
      </header>

      <main className={styles.main}>
        <textarea className={styles.textarea} value={savedText} readOnly />
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
  savedText: PropTypes.string.isRequired,
  onBack: PropTypes.func.isRequired,
  onFinish: PropTypes.func.isRequired,
};

export default AnnotationReviewPage;
