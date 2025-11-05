import PropTypes from 'prop-types';
import styles from '../AnnotationCanvas.module.css';

const HIGHLIGHT_COLOR = '#a855f7';

const LineAnnotation = ({
  line,
  label,
  isSelected,
  isHighlighted = false,
  imageBox,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onHandlePointerDown,
  onHandlePointerMove,
}) => {
  const baseStroke = label?.color || '#f59e0b';
  const stroke = baseStroke;
  const startX = line.x1 * imageBox.width;
  const startY = line.y1 * imageBox.height;
  const endX = line.x2 * imageBox.width;
  const endY = line.y2 * imageBox.height;

  return (
    <g className={styles.lineGroup}>
      {/* 1. 보이는 선 (이벤트 받지 않음) */}
      <line
        x1={startX}
        y1={startY}
        x2={endX}
        y2={endY}
        stroke={stroke}
        strokeWidth={2} // 두께는 2px로 유지 (BoxAnnotation.jsx와 일치)
        strokeLinecap='round'
        pointerEvents='none'
      />
      {/* 2. 투명한 히트박스 (클릭 영역) */}
      <line
        x1={startX}
        y1={startY}
        x2={endX}
        y2={endY}
        stroke='transparent'
        strokeWidth={16}
        strokeLinecap='round'
        className={styles.line}
        onPointerDown={(event) => onPointerDown(event, line)}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      />
      {isHighlighted && (
        <line
          x1={startX}
          y1={startY}
          x2={endX}
          y2={endY}
          stroke={HIGHLIGHT_COLOR}
          strokeWidth={8}
          strokeLinecap='round'
          strokeOpacity={0.4}
          strokeDasharray={null}
          pointerEvents='none'
        />
      )}
      {isSelected && (
        <>
          <circle
            cx={startX}
            cy={startY}
            r={4}
            className={styles.lineHandle}
            onPointerDown={(event) => onHandlePointerDown(event, line, 'start')}
            onPointerMove={onHandlePointerMove}
            onPointerUp={onPointerUp}
            style={isHighlighted ? { stroke: HIGHLIGHT_COLOR } : undefined}
          />
          <circle
            cx={endX}
            cy={endY}
            r={4}
            className={styles.lineHandle}
            onPointerDown={(event) => onHandlePointerDown(event, line, 'end')}
            onPointerMove={onHandlePointerMove}
            onPointerUp={onPointerUp}
            style={isHighlighted ? { stroke: HIGHLIGHT_COLOR } : undefined}
          />
        </>
      )}
    </g>
  );
};

LineAnnotation.propTypes = {
  line: PropTypes.shape({
    id: PropTypes.string.isRequired,
    labelId: PropTypes.string.isRequired,
    x1: PropTypes.number.isRequired,
    y1: PropTypes.number.isRequired,
    x2: PropTypes.number.isRequired,
    y2: PropTypes.number.isRequired,
  }).isRequired,
  label: PropTypes.shape({
    name: PropTypes.string,
    color: PropTypes.string,
  }),
  isSelected: PropTypes.bool.isRequired,
  isHighlighted: PropTypes.bool,
  imageBox: PropTypes.shape({
    width: PropTypes.number.isRequired,
    height: PropTypes.number.isRequired,
  }).isRequired,
  onPointerDown: PropTypes.func.isRequired,
  onPointerMove: PropTypes.func.isRequired,
  onPointerUp: PropTypes.func.isRequired,
  onHandlePointerDown: PropTypes.func.isRequired,
  onHandlePointerMove: PropTypes.func.isRequired,
};

LineAnnotation.defaultProps = {
  label: undefined,
  isHighlighted: false,
};

export default LineAnnotation;
