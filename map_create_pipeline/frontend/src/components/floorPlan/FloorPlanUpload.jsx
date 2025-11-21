import { useMemo, useRef, useState } from 'react';
import PropTypes from 'prop-types';
import { Image as ImageIcon, Loader2, Upload } from 'lucide-react';
import styles from './FloorPlanUpload.module.css';

const FLOOR_TYPE_OPTIONS = [
  { value: 'B', label: '지하' },
  { value: 'F', label: '지상' },
];

const FloorPlanUpload = ({ onSelectFile, isUploading, error, existingFloors }) => {
  const inputRef = useRef(null);
  const [floorType, setFloorType] = useState('');
  const [floorNumber, setFloorNumber] = useState('');

  const sanitisedNumber = useMemo(() => floorNumber.replace(/\D/g, ''), [floorNumber]);

  const resolvedFloorLabel = useMemo(() => {
    if (!floorType || !sanitisedNumber) {
      return '';
    }
    if (floorType === 'B') {
      return `지하 ${sanitisedNumber}층`;
    }
    return `지상 ${sanitisedNumber}층`;
  }, [floorType, sanitisedNumber]);

  const resolvedFloorValue = useMemo(() => {
    if (!floorType || !sanitisedNumber) {
      return '';
    }
    return `${floorType}${sanitisedNumber}`;
  }, [floorType, sanitisedNumber]);

  const existingFloorTokens = useMemo(() => {
    const tokens = new Set();
    (existingFloors || []).forEach((entry) => {
      if (!entry) {
        return;
      }
      const value = (entry.floorValue || '').trim().toUpperCase();
      const label = (entry.floorLabel || '').trim();
      if (value) {
        tokens.add(value);
      }
      if (label) {
        tokens.add(`LABEL:${label}`);
      }
    });
    return tokens;
  }, [existingFloors]);

  const currentFloorTokens = useMemo(() => {
    const tokens = [];
    if (resolvedFloorValue) {
      tokens.push(resolvedFloorValue.trim().toUpperCase());
    }
    if (resolvedFloorLabel) {
      tokens.push(`LABEL:${resolvedFloorLabel.trim()}`);
    }
    return tokens;
  }, [resolvedFloorLabel, resolvedFloorValue]);

  const isDuplicateFloor = useMemo(() => {
    if (!currentFloorTokens.length) {
      return false;
    }
    for (const token of currentFloorTokens) {
      if (existingFloorTokens.has(token)) {
        return true;
      }
    }
    return false;
  }, [currentFloorTokens, existingFloorTokens]);

  const handleClick = () => {
    if (!resolvedFloorLabel || !resolvedFloorValue) {
      // eslint-disable-next-line no-alert
      alert('이미지를 선택하기 전에 층 정보를 먼저 선택해 주세요.');
      return;
    }
    if (isDuplicateFloor) {
      // eslint-disable-next-line no-alert
      alert('이미 존재하는 층입니다. 다른 층을 선택해 주세요.');
      return;
    }
    inputRef.current?.click();
  };

  const handleChange = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!resolvedFloorLabel || !resolvedFloorValue) {
        // eslint-disable-next-line no-alert
        alert('층 정보가 선택되지 않았습니다. 다시 시도해 주세요.');
        return;
      }
      if (isDuplicateFloor) {
        // eslint-disable-next-line no-alert
        alert('이미 존재하는 층입니다. 다른 층을 선택해 주세요.');
        return;
      }
      onSelectFile?.({
        file,
        floorLabel: resolvedFloorLabel,
        floorValue: resolvedFloorValue,
      });
      event.target.value = '';
    }
  };

  return (
    <div className={styles.card}>
      <div className={styles.iconWrapper}>
        <ImageIcon size={36} />
      </div>
      <h2 className={styles.title}>도면 이미지 업로드</h2>
      <p className={styles.description}>
        객체 감지 분석을 위해 도면 이미지를 업로드하세요. 서버에서 자동으로 객체를 감지한 후 수정할 수 있습니다.
      </p>
      <div className={styles.floorSelector}>
        <label htmlFor='floor-type'>도면 층 선택</label>
        <div className={styles.floorRow}>
          <select
            id='floor-type'
            className={styles.floorSelect}
            value={floorType}
            onChange={(event) => setFloorType(event.target.value)}
            disabled={isUploading}
          >
            <option value=''>지하/지상 선택</option>
            {FLOOR_TYPE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <input
            type='text'
            className={styles.floorNumberInput}
            placeholder='층수'
            value={sanitisedNumber}
            onChange={(event) => setFloorNumber(event.target.value.replace(/\D/g, ''))}
            inputMode='numeric'
            pattern='[0-9]*'
            maxLength={2}
            disabled={isUploading}
          />
          <span className={styles.floorSuffix}>층</span>
        </div>
        <p className={`${styles.floorHint} ${isDuplicateFloor ? styles.floorHintWarning : ''}`}>
          {isDuplicateFloor
            ? '이미 존재하는 층입니다. 다른 층을 선택해 주세요.'
            : resolvedFloorLabel
              ? `선택된 층: ${resolvedFloorLabel}`
              : '먼저 층 정보를 선택한 뒤 이미지를 업로드하세요.'}
        </p>
      </div>
      <input
        ref={inputRef}
        type='file'
        accept='image/*'
        className={styles.hiddenInput}
        onChange={handleChange}
        disabled={isUploading}
      />
      <button
        type='button'
        className={styles.button}
        onClick={handleClick}
        disabled={isUploading || !resolvedFloorLabel || !resolvedFloorValue || isDuplicateFloor}
      >
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
  existingFloors: PropTypes.arrayOf(
    PropTypes.shape({
      floorLabel: PropTypes.string,
      floorValue: PropTypes.string,
    })
  ),
};

FloorPlanUpload.defaultProps = {
  onSelectFile: undefined,
  isUploading: false,
  error: null,
  existingFloors: [],
};

export default FloorPlanUpload;
