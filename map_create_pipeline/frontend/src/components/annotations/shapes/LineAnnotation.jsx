import PropTypes from 'prop-types';
import styles from '../AnnotationCanvas.module.css';

const DEFAULT_HIGHLIGHT_COLOR = '#a855f7';

const HANDLE_SCREEN_RADIUS = 4;
const DEFAULT_STROKE_WIDTH = 2;
const DEFAULT_HIT_WIDTH = 16;

const LineAnnotation = ({
  line,
  label,
  isSelected,
  isHighlighted = false,
  imageBox,
  viewportScale = 1,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onHandlePointerDown,
  onHandlePointerMove,
  strokeWidth = DEFAULT_STROKE_WIDTH,
  hitStrokeWidth = DEFAULT_HIT_WIDTH,
  handleRadius: handleRadiusInput,
  highlightColor = DEFAULT_HIGHLIGHT_COLOR,
}) => {
  const baseStroke = label?.color || '#f59e0b';
  const stroke = baseStroke;
  const startX = line.x1 * imageBox.width;
  const startY = line.y1 * imageBox.height;
  const endX = line.x2 * imageBox.width;
  const endY = line.y2 * imageBox.height;
  const safeScale = Number.isFinite(viewportScale) && viewportScale > 0 ? viewportScale : 1;
  const resolvedHandleBase = Number.isFinite(handleRadiusInput) && handleRadiusInput > 0 ? handleRadiusInput : HANDLE_SCREEN_RADIUS;
  const handleRadius = resolvedHandleBase / safeScale;

  return (
    <g className={styles.lineGroup}>
      {/* 1. 보이는 선 (이벤트 받지 않음) */}
      <line
        x1={startX}
        y1={startY}
        x2={endX}
        y2={endY}
        stroke={stroke}
        strokeWidth={strokeWidth}
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
        strokeWidth={hitStrokeWidth}
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
          stroke={highlightColor}
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
            r={handleRadius}
            className={styles.lineHandle}
            onPointerDown={(event) => onHandlePointerDown(event, line, 'start')}
            onPointerMove={onHandlePointerMove}
            onPointerUp={onPointerUp}
            style={isHighlighted ? { stroke: highlightColor } : undefined}
          />
          <circle
            cx={endX}
            cy={endY}
            r={handleRadius}
            className={styles.lineHandle}
            onPointerDown={(event) => onHandlePointerDown(event, line, 'end')}
            onPointerMove={onHandlePointerMove}
            onPointerUp={onPointerUp}
            style={isHighlighted ? { stroke: highlightColor } : undefined}
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
  viewportScale: PropTypes.number,
  onPointerDown: PropTypes.func.isRequired,
  onPointerMove: PropTypes.func.isRequired,
  onPointerUp: PropTypes.func.isRequired,
  onHandlePointerDown: PropTypes.func.isRequired,
  onHandlePointerMove: PropTypes.func.isRequired,
  strokeWidth: PropTypes.number,
  hitStrokeWidth: PropTypes.number,
  handleRadius: PropTypes.number,
  highlightColor: PropTypes.string,
};

LineAnnotation.defaultProps = {
  label: undefined,
  isHighlighted: false,
  viewportScale: 1,
  strokeWidth: DEFAULT_STROKE_WIDTH,
  hitStrokeWidth: DEFAULT_HIT_WIDTH,
  handleRadius: undefined,
  highlightColor: DEFAULT_HIGHLIGHT_COLOR,
};

export default LineAnnotation;
