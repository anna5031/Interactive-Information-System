import PropTypes from 'prop-types';
import styles from '../AnnotationCanvas.module.css';

const LineAnnotation = ({
  line,
  label,
  isSelected,
  imageBox,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onHandlePointerDown,
  onHandlePointerMove,
}) => {
  const stroke = label?.color || '#f59e0b';
  const startX = line.x1 * imageBox.width;
  const startY = line.y1 * imageBox.height;
  const endX = line.x2 * imageBox.width;
  const endY = line.y2 * imageBox.height;

  return (
    <g className={styles.lineGroup}>
      <line
        x1={startX}
        y1={startY}
        x2={endX}
        y2={endY}
        stroke={stroke}
        strokeWidth={4}
        strokeLinecap='round'
        className={styles.line}
        onPointerDown={(event) => onPointerDown(event, line)}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      />
      {isSelected && (
        <>
          <circle
            cx={startX}
            cy={startY}
            r={8}
            className={styles.lineHandle}
            onPointerDown={(event) => onHandlePointerDown(event, line, 'start')}
            onPointerMove={onHandlePointerMove}
            onPointerUp={onPointerUp}
          />
          <circle
            cx={endX}
            cy={endY}
            r={8}
            className={styles.lineHandle}
            onPointerDown={(event) => onHandlePointerDown(event, line, 'end')}
            onPointerMove={onHandlePointerMove}
            onPointerUp={onPointerUp}
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
};

export default LineAnnotation;
