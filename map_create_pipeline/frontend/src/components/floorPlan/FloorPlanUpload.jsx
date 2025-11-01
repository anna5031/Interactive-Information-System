import { useRef } from 'react';
import PropTypes from 'prop-types';
import { Image as ImageIcon, Loader2, Upload } from 'lucide-react';
import styles from './FloorPlanUpload.module.css';

const FloorPlanUpload = ({ onSelectFile, isUploading, error }) => {
  const inputRef = useRef(null);

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleChange = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      onSelectFile?.(file);
    }
  };

  return (
    <div className={styles.card}>
      <div className={styles.iconWrapper}>
        <ImageIcon size={36} />
      </div>
      <h2 className={styles.title}>도면 이미지 업로드</h2>
      <p className={styles.description}>
        YOLO 분석을 위해 도면 이미지를 업로드하세요. 서버에서 자동으로 객체를 감지한 후 수정할 수 있습니다.
      </p>
      <input
        ref={inputRef}
        type='file'
        accept='image/*'
        className={styles.hiddenInput}
        onChange={handleChange}
        disabled={isUploading}
      />
      <button type='button' className={styles.button} onClick={handleClick} disabled={isUploading}>
        {isUploading ? <Loader2 className={styles.spinner} size={18} /> : <Upload size={18} />}
        {isUploading ? '분석 중...' : '이미지 선택하기'}
      </button>
      {error && <p className={styles.error}>{error}</p>}
    </div>
  );
};

FloorPlanUpload.propTypes = {
  onSelectFile: PropTypes.func,
  isUploading: PropTypes.bool,
  error: PropTypes.string,
};

FloorPlanUpload.defaultProps = {
  onSelectFile: undefined,
  isUploading: false,
  error: null,
};

export default FloorPlanUpload;
