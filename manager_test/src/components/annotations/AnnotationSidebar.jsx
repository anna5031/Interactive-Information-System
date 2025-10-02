import PropTypes from 'prop-types';
import { Trash2, SquarePlus } from 'lucide-react';
import LABEL_CONFIG, { getLabelById, isLineLabel, isBoxLabel } from '../../config/annotationConfig';
import styles from './AnnotationSidebar.module.css';

const formatPercentage = (value) => `${(value * 100).toFixed(1)}%`;

const formatLineLength = (line) => {
  const length = Math.hypot(line.x2 - line.x1, line.y2 - line.y1);
  return `${(length * 100).toFixed(1)}%`;
};

const AnnotationSidebar = ({
  boxes,
  lines,
  selectedBox,
  selectedLine,
  onSelect,
  onDelete,
  onLabelChange,
  activeLabelId,
  onActiveLabelChange,
  addMode,
  onToggleAddMode,
  hiddenLabelIds,
  onToggleLabelVisibility,
  isLineLabelActive,
}) => {
  const isBoxAddDisabled = addMode && !isBoxLabel(activeLabelId);
  const isLineAddDisabled = addMode && !isLineLabel(activeLabelId);

  const handleSelectBox = (box) => onSelect?.({ type: 'box', id: box.id });
  const handleSelectLine = (line) => onSelect?.({ type: 'line', id: line.id });

  const handleLabelChangeBox = (event) => {
    if (selectedBox) {
      onLabelChange?.({ type: 'box', id: selectedBox.id }, event.target.value);
    }
  };

  const handleLabelChangeLine = (event) => {
    if (selectedLine) {
      onLabelChange?.({ type: 'line', id: selectedLine.id }, event.target.value);
    }
  };

  return (
    <aside className={styles.sidebar}>
      <div className={styles.section}>
        <h3 className={styles.heading}>라벨 추가</h3>
        <div className={styles.fieldGroup}>
          <label className={styles.label} htmlFor='label-selector'>
            객체 종류
          </label>
          <select
            id='label-selector'
            className={styles.select}
            value={activeLabelId}
            onChange={(event) => onActiveLabelChange?.(event.target.value)}
          >
            {LABEL_CONFIG.map((label) => (
              <option key={label.id} value={label.id}>
                {`${label.id} - ${label.name}`}
              </option>
            ))}
          </select>
        </div>
        <button
          type='button'
          className={`${styles.button} ${addMode ? styles.buttonSecondary : ''}`}
          onClick={onToggleAddMode}
        >
          <SquarePlus size={16} />
          {addMode ? '추가 종료' : '새 객체 추가'}
        </button>
        {addMode && (
          <p className={styles.helperText}>
            {isLineLabelActive
              ? '화면에서 선을 그리고 놓으면 선이 추가됩니다.'
              : '화면에서 드래그하여 박스를 추가하세요.'}
          </p>
        )}
      </div>

      <div className={styles.section}>
        <h3 className={styles.heading}>라벨 표시</h3>
        <div className={styles.filterList}>
          {LABEL_CONFIG.map((label) => {
            const hidden = hiddenLabelIds?.has(label.id);
            return (
              <button
                key={label.id}
                type='button'
                className={`${styles.filterTag} ${hidden ? styles.filterTagInactive : ''}`}
                onClick={() => onToggleLabelVisibility?.(label.id)}
              >
                <span className={styles.colorIndicator} style={{ backgroundColor: label.color }} />
                {label.name}
                {hidden && <span className={styles.hiddenBadge}>숨김</span>}
              </button>
            );
          })}
        </div>
      </div>

      <div className={styles.section}>
        <h3 className={styles.heading}>박스 객체</h3>
        <ul className={styles.list}>
          {boxes.map((box) => {
            const label = getLabelById(box.labelId) || LABEL_CONFIG[0];
            const isSelected = selectedBox?.id === box.id;
            return (
              <li key={box.id}>
                <button
                  type='button'
                  className={`${styles.listItem} ${isSelected ? styles.listItemActive : ''}`}
                  onClick={() => handleSelectBox(box)}
                  disabled={isLineLabelActive}
                >
                  <span className={styles.colorIndicator} style={{ backgroundColor: label?.color }} />
                  <span className={styles.listText}>{`${label?.name ?? '라벨'} (${box.labelId})`}</span>
                  <span className={styles.dimensions}>
                    {formatPercentage(box.width)} × {formatPercentage(box.height)}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      <div className={styles.section}>
        <h3 className={styles.heading}>벽(선) 객체</h3>
        <ul className={styles.list}>
          {lines.map((line) => {
            const label = getLabelById(line.labelId) || LABEL_CONFIG[0];
            const isSelected = selectedLine?.id === line.id;
            return (
              <li key={line.id}>
                <button
                  type='button'
                  className={`${styles.listItem} ${isSelected ? styles.listItemActive : ''}`}
                  onClick={() => handleSelectLine(line)}
                  disabled={!isLineLabelActive && addMode}
                >
                  <span className={styles.colorIndicator} style={{ backgroundColor: label?.color }} />
                  <span className={styles.listText}>{`${label?.name ?? '벽'} (${line.labelId})`}</span>
                  <span className={styles.dimensions}>{formatLineLength(line)}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      {selectedBox && (
        <div className={styles.section}>
          <h3 className={styles.heading}>선택된 박스</h3>
          <div className={styles.fieldGroup}>
            <label className={styles.label} htmlFor='selected-box-label'>
              라벨
            </label>
            <select
              id='selected-box-label'
              className={styles.select}
              value={selectedBox.labelId}
              onChange={handleLabelChangeBox}
            >
              {LABEL_CONFIG.filter((label) => isBoxLabel(label.id)).map((label) => (
                <option key={label.id} value={label.id}>
                  {`${label.id} - ${label.name}`}
                </option>
              ))}
            </select>
          </div>
          <div className={styles.stats}>
            <div>
              <span className={styles.statLabel}>위치</span>
              <span className={styles.statValue}>
                {formatPercentage(selectedBox.x)}, {formatPercentage(selectedBox.y)}
              </span>
            </div>
            <div>
              <span className={styles.statLabel}>크기</span>
              <span className={styles.statValue}>
                {formatPercentage(selectedBox.width)} × {formatPercentage(selectedBox.height)}
              </span>
            </div>
          </div>
          <button
            type='button'
            className={`${styles.button} ${styles.danger}`}
            onClick={() => onDelete?.({ type: 'box', id: selectedBox.id })}
          >
            <Trash2 size={16} />
            삭제하기
          </button>
        </div>
      )}

      {selectedLine && (
        <div className={styles.section}>
          <h3 className={styles.heading}>선택된 선</h3>
          <div className={styles.fieldGroup}>
            <label className={styles.label} htmlFor='selected-line-label'>
              라벨
            </label>
            <select
              id='selected-line-label'
              className={styles.select}
              value={selectedLine.labelId}
              onChange={handleLabelChangeLine}
            >
              {LABEL_CONFIG.filter((label) => isLineLabel(label.id)).map((label) => (
                <option key={label.id} value={label.id}>
                  {`${label.id} - ${label.name}`}
                </option>
              ))}
            </select>
          </div>
          <div className={styles.stats}>
            <div>
              <span className={styles.statLabel}>시작</span>
              <span className={styles.statValue}>
                {formatPercentage(selectedLine.x1)}, {formatPercentage(selectedLine.y1)}
              </span>
            </div>
            <div>
              <span className={styles.statLabel}>끝</span>
              <span className={styles.statValue}>
                {formatPercentage(selectedLine.x2)}, {formatPercentage(selectedLine.y2)}
              </span>
            </div>
            <div>
              <span className={styles.statLabel}>길이</span>
              <span className={styles.statValue}>{formatLineLength(selectedLine)}</span>
            </div>
          </div>
          <button
            type='button'
            className={`${styles.button} ${styles.danger}`}
            onClick={() => onDelete?.({ type: 'line', id: selectedLine.id })}
          >
            <Trash2 size={16} />
            삭제하기
          </button>
        </div>
      )}
    </aside>
  );
};

AnnotationSidebar.propTypes = {
  boxes: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      labelId: PropTypes.string.isRequired,
      x: PropTypes.number.isRequired,
      y: PropTypes.number.isRequired,
      width: PropTypes.number.isRequired,
      height: PropTypes.number.isRequired,
    })
  ).isRequired,
  lines: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      labelId: PropTypes.string.isRequired,
      x1: PropTypes.number.isRequired,
      y1: PropTypes.number.isRequired,
      x2: PropTypes.number.isRequired,
      y2: PropTypes.number.isRequired,
    })
  ).isRequired,
  selectedBox: PropTypes.shape({
    id: PropTypes.string.isRequired,
    labelId: PropTypes.string.isRequired,
    x: PropTypes.number.isRequired,
    y: PropTypes.number.isRequired,
    width: PropTypes.number.isRequired,
    height: PropTypes.number.isRequired,
  }),
  selectedLine: PropTypes.shape({
    id: PropTypes.string.isRequired,
    labelId: PropTypes.string.isRequired,
    x1: PropTypes.number.isRequired,
    y1: PropTypes.number.isRequired,
    x2: PropTypes.number.isRequired,
    y2: PropTypes.number.isRequired,
  }),
  onSelect: PropTypes.func,
  onDelete: PropTypes.func,
  onLabelChange: PropTypes.func,
  activeLabelId: PropTypes.string.isRequired,
  onActiveLabelChange: PropTypes.func,
  addMode: PropTypes.bool,
  onToggleAddMode: PropTypes.func,
  hiddenLabelIds: PropTypes.instanceOf(Set),
  onToggleLabelVisibility: PropTypes.func,
  isLineLabelActive: PropTypes.bool,
};

AnnotationSidebar.defaultProps = {
  selectedBox: null,
  selectedLine: null,
  onSelect: undefined,
  onDelete: undefined,
  onLabelChange: undefined,
  onActiveLabelChange: undefined,
  addMode: false,
  onToggleAddMode: undefined,
  hiddenLabelIds: undefined,
  onToggleLabelVisibility: undefined,
  isLineLabelActive: false,
};

export default AnnotationSidebar;
